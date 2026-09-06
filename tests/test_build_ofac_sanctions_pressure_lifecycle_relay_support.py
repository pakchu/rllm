import numpy as np,pandas as pd
from training import build_ofac_sanctions_pressure_lifecycle_relay_support as support

def test_taxonomy_xor():
 assert support.classify("Iran-related Designations","")==-1
 assert support.classify("Sanctions List Removals","")==0
 assert support.classify("General License","")==1
 assert support.classify("Designations and General License","")==0
 assert support.classify("Frequently Asked Questions","")==0

def test_parser_reads_official_listing_shape():
 raw=b'''<div class="margin-bottom-4 search-result views-row"><div><a href="/recent-actions/20250701">Iran-related Designations</a></div><div>July 01, 2025 - <a href="/recent-actions/sanctions-list-updates">Sanctions List Updates</a></div></div>'''
 rows=support.parse_page(raw);assert len(rows)==1;assert rows[0]["title"]=="iran-related designations";assert rows[0]["summary"]=="sanctions list updates"

def test_rank_excludes_current():
 r=support.rank(pd.Series(np.arange(127,dtype=float)));assert np.isnan(r.iloc[125]);assert r.iloc[126]==1.
