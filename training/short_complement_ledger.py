"""Portfolio-complement shorts: reviewed net ledger with candidate-specific last sleeve.

Hedge quantity is capped by observable positive parent net units and is resized
on parent/hedge events; no new net short is created by this overlay.
"""
from itertools import permutations
import numpy as np
import pandas as pd


def simulate(data, targets, events, barrier_prices, weights, names, cost=.0006, net_cap=4.5, hedge_targets=None, hedge_events=None):
    targets=np.asarray(targets,float);events=np.asarray(events,bool)
    weights=np.asarray(weights,float);barrier_prices=np.asarray(barrier_prices,float)
    n,s=targets.shape;k=len(weights)
    if events.shape!=(n,s) or barrier_prices.shape!=(n,s) or weights.shape!=(k,s):raise ValueError('shape mismatch')
    if not np.isfinite(targets).all() or not np.isfinite(weights).all() or (weights<0).any():raise ValueError('invalid targets/weights')
    if hedge_targets is None or hedge_events is None:raise ValueError('hedge inputs required')
    hedge_targets=np.asarray(hedge_targets,float);hedge_events=np.asarray(hedge_events,bool)
    if hedge_targets.shape!=(n,k) or hedge_events.shape!=(n,k) or not np.isfinite(hedge_targets).all() or (hedge_targets>0).any():raise ValueError('invalid short hedge')
    if len(data['open'])!=n or n==0:raise ValueError('empty or mismatched market')
    eq=np.ones(k);peak=eq.copy();dd=np.zeros(k);units=np.zeros((k,s))
    fees=np.zeros(k);funding=np.zeros(k);orders=np.zeros(k,int);entries=np.zeros(k,int)
    hedge_entries=np.zeros(k,int);hedge_hours=np.zeros(k)
    turnover=np.zeros(k);cap_hits=np.zeros(k,int);dead=np.zeros(k,bool)
    paths=np.zeros((n,k));maxnet=np.zeros(k);mean_net=np.zeros(k)
    for t in range(n):
        op=data['open'][t];previous=units.sum(axis=1);prior=eq.copy()
        ev=np.broadcast_to(events[t],(k,s)).copy();ev[:,-1]=hedge_events[t]
        parent_event=(ev[:,:-1] & (weights[:,:-1]>0)).any(axis=1)
        ev[:,-1]|=parent_event
        updated=np.where(ev,weights*targets[t]*eq[:,None]/op,units)
        parent_net=updated[:,:-1].sum(axis=1)
        wanted=weights[:,-1]*hedge_targets[t]*eq/op
        capped=-np.minimum(np.maximum(parent_net,0),np.maximum(-wanted,0))
        # Known parent events can close/reopen or resize a currently active hedge.
        updated[:,-1]=np.where(ev[:,-1],capped,units[:,-1])
        hedge_entries+=(units[:,-1]>=-1e-15)&(updated[:,-1]<-1e-15)
        hedge_hours+=(updated[:,-1]<-1e-15)/12
        active=(ev & (weights>0)).any(axis=1)
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
            for sequence in permutations(np.unique(prices[closed])):
                scenario_units=units.copy();cash=bar_cash.copy()
                for price in sequence:
                    group=closed[prices[closed]==price]
                    closing=scenario_units[:,group].sum(axis=1)
                    parent_after=scenario_units[:,:-1].sum(axis=1)-closing
                    cover=np.maximum(-scenario_units[:,-1]-np.maximum(parent_after,0),0)
                    closing-=cover
                    cash+=closing*(price-op)-np.abs(closing)*price*cost
                    scenario_units[:,group]=0
                    scenario_units[:,-1]+=cover
                remaining=scenario_units.sum(axis=1)
                scenario_high=cash+remaining*(np.where(remaining>=0,data['high'][t],data['low'][t])-op)
                scenario_low=cash+remaining*(np.where(remaining>=0,data['low'][t],data['high'][t])-op)
                intrabar_ruin|=scenario_low<=0
                peak=np.maximum(peak,scenario_high);dd=np.maximum(dd,1-scenario_low/np.maximum(peak,1e-12))
        if len(barrier_sleeves):
            # Unknown same-bar ordering: retain the lowest terminal equity path.
            best=None
            for sequence in permutations(np.unique(prices[barrier_sleeves])):
                trial_units=units.copy();trial_eq=eq.copy()
                trial_fees=np.zeros(k);trial_orders=np.zeros(k,int);trial_turnover=np.zeros(k)
                for price in sequence:
                    close=prices==price
                    closing=trial_units[:,close].sum(axis=1)
                    parent_after=trial_units[:,:-1].sum(axis=1)-closing
                    cover=np.maximum(-trial_units[:,-1]-np.maximum(parent_after,0),0)
                    closing-=cover
                    amount=np.abs(closing)*price;charge=amount*cost
                    trial_eq+=closing*(price-end)-charge
                    trial_fees+=charge;trial_orders+=amount>1e-10
                    trial_turnover+=amount/np.maximum(prior,1e-12)
                    trial_units[:,close]=0;trial_units[:,-1]+=cover
                if best is None:
                    best=[trial_eq,trial_units,trial_fees,trial_orders,trial_turnover]
                else:
                    lower=trial_eq<best[0]
                    for old,new in zip(best,[trial_eq,trial_units,trial_fees,trial_orders,trial_turnover]):old[lower]=new[lower]
            eq,units,added_fees,added_orders,added_turnover=best
            fees+=added_fees;orders+=added_orders;turnover+=added_turnover
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
                     'entry_episodes':int(entries[i]),'hedge_entries':int(hedge_entries[i]),'hedge_hours':float(hedge_hours[i]),'orders':int(orders[i]),'turnover':float(turnover[i]),
                     'fees_pct_initial':float(fees[i]*100),'funding_pct_initial':float(funding[i]*100),
                     'max_open_net_exposure':float(maxnet[i]),'mean_open_net_exposure':float(mean_net[i]/n),
                     'cap_interventions':int(cap_hits[i]),'insolvent':bool(dead[i])})
    return rows,paths
