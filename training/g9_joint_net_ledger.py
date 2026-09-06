"""Shared BTC fixed-unit ledger for native events and intrabar barrier exits.

Signed sleeve units offset before open-event costs/funding. Barrier exits execute
at their own prices; simultaneous same-price exits net. Portfolio cap intervention
may resize carried sleeves at active events. This is research, not a liquidation
or market-impact model.
"""
import numpy as np
import pandas as pd


def simulate(data, targets, events, barrier_prices, weights, names, cost=.0006, net_cap=4.5):
    targets=np.asarray(targets,float);events=np.asarray(events,bool)
    weights=np.asarray(weights,float);barrier_prices=np.asarray(barrier_prices,float)
    n,s=targets.shape;k=len(weights)
    if events.shape!=(n,s) or barrier_prices.shape!=(n,s) or weights.shape!=(k,s):raise ValueError('shape mismatch')
    if not np.isfinite(targets).all() or not np.isfinite(weights).all() or (weights<0).any():raise ValueError('invalid targets/weights')
    if len(data['open'])!=n or n==0:raise ValueError('empty or mismatched market')
    eq=np.ones(k);peak=eq.copy();dd=np.zeros(k);units=np.zeros((k,s))
    fees=np.zeros(k);funding=np.zeros(k);orders=np.zeros(k,int);entries=np.zeros(k,int)
    turnover=np.zeros(k);cap_hits=np.zeros(k,int);dead=np.zeros(k,bool)
    paths=np.zeros((n,k));maxnet=np.zeros(k);mean_net=np.zeros(k)
    for t in range(n):
        op=data['open'][t];previous=units.sum(axis=1);prior=eq.copy()
        updated=np.where(events[t],weights*targets[t]*eq[:,None]/op,units)
        active=(events[t]& (weights>0)).any(axis=1)
        signed=updated.sum(axis=1);exposure=np.abs(signed*op)
        # Solve T <= cap * (equity - fee*abs(T - aligned_prior)) exactly.
        aligned_prior=np.sign(signed)*previous*op
        budget=np.maximum(eq,0)*net_cap; fee_cap=net_cap*cost
        if fee_cap>=1:raise ValueError('cost times cap must be below one')
        limit=np.where(budget>=aligned_prior,(budget+fee_cap*aligned_prior)/(1+fee_cap),
                       (budget-fee_cap*aligned_prior)/(1-fee_cap))
        scale=np.minimum(1,np.maximum(limit,0)/np.maximum(exposure,1e-15))
        cap_hits+=(scale<1-1e-12)&active&~dead
        units=updated*np.where(active,scale,1)[:,None];units[dead]=0
        net=units.sum(axis=1);delta=(net-previous)*op
        charge=np.abs(delta)*cost;fees+=charge;eq-=charge
        orders+=np.abs(delta)>1e-10
        entries+=(net*previous<=0)&(np.abs(net)>1e-15)
        turnover+=np.abs(delta)/np.maximum(prior,1e-12)
        risk=np.abs(net*op)/np.maximum(eq,1e-12);maxnet=np.maximum(maxnet,risk);mean_net+=risk
        transfer=net*data['funding'][t];funding+=transfer
        high=eq+net*(np.where(net>=0,data['high'][t],data['low'][t])-op)+np.maximum(-transfer,0)
        low=eq+net*(np.where(net>=0,data['low'][t],data['high'][t])-op)-np.maximum(transfer,0)
        intrabar_ruin=low<=0
        peak=np.maximum(peak,high);dd=np.maximum(dd,1-low/np.maximum(peak,1e-12))
        bar_cash=eq-transfer
        # Start with all units marked to next open, then correct intrabar exits.
        end=data['end'][t];eq+=net*(end-op)-transfer
        prices=barrier_prices[t]
        barrier_sleeves=np.flatnonzero(np.isfinite(prices))
        # Every subset of intrabar exits is possible without a tick-order feed.
        # Include remaining unhedged exposure after one side of a hedge exits.
        for subset in range(1, 1 << len(barrier_sleeves)):
            closed=barrier_sleeves[[bool(subset & (1 << j)) for j in range(len(barrier_sleeves))]]
            remaining=net-units[:,closed].sum(axis=1)
            realized=(units[:,closed]*(prices[closed]-op)).sum(axis=1)
            exit_fee=np.zeros(k)
            for price in np.unique(prices[closed]):
                group=closed[prices[closed]==price]
                exit_fee+=np.abs(units[:,group].sum(axis=1))*price*cost
            cash=bar_cash+realized-exit_fee
            scenario_high=cash+remaining*(np.where(remaining>=0,data['high'][t],data['low'][t])-op)
            scenario_low=cash+remaining*(np.where(remaining>=0,data['low'][t],data['high'][t])-op)
            intrabar_ruin|=scenario_low<=0
            peak=np.maximum(peak,scenario_high);dd=np.maximum(dd,1-scenario_low/np.maximum(peak,1e-12))
        for price in np.unique(prices[np.isfinite(prices)]):
            close=prices==price
            closing_units=units[:,close].sum(axis=1)
            eq+=closing_units*(price-end)
            charge=np.abs(closing_units)*price*cost;fees+=charge;eq-=charge
            orders+=np.abs(closing_units)*price>1e-10
            turnover+=np.abs(closing_units)*price/np.maximum(prior,1e-12)
            units[:,close]=0
        if t==n-1:
            liquidation=np.abs(units.sum(axis=1))*end
            fees+=liquidation*cost;eq-=liquidation*cost
            orders+=liquidation>1e-10;turnover+=liquidation/np.maximum(prior,1e-12)
            units[:]=0
        dead|=(eq<=0)|intrabar_ruin;eq=np.where(dead,0,eq);units[dead]=0
        dd=np.minimum(dd,1)
        peak=np.maximum(peak,eq);dd=np.maximum(dd,1-eq/np.maximum(peak,1e-12))
        paths[t]=eq
    years=(pd.Timestamp(data['end_date'][-1])-pd.Timestamp(data['date'][0])).total_seconds()/(365.25*86400)
    cagr=(eq**(1/years)-1)*100
    rows=[]
    for i,w in enumerate(weights):
        rows.append({'weights_notional':dict(zip(names,map(float,w))), 'return_pct':float((eq[i]-1)*100),
                     'cagr_pct':float(cagr[i]),'mdd_pct':float(dd[i]*100),
                     'calmar':float(cagr[i]/(dd[i]*100)) if dd[i]>1e-12 else 0.,
                     'entry_episodes':int(entries[i]),'orders':int(orders[i]),'turnover':float(turnover[i]),
                     'fees_pct_initial':float(fees[i]*100),'funding_pct_initial':float(funding[i]*100),
                     'max_open_net_exposure':float(maxnet[i]),'mean_open_net_exposure':float(mean_net[i]/n),
                     'cap_interventions':int(cap_hits[i]),'insolvent':bool(dead[i])})
    return rows,paths
