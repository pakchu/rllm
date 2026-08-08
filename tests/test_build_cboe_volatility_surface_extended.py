import csv,io
from training import build_cboe_volatility_surface_extended as builder
def test_parse_supports_two_column_and_ohlc_cboe_schemas():
 tail=b"DATE,SKEW\n07/31/2026,145.2\n";ohlc=b"DATE,OPEN,HIGH,LOW,CLOSE\n07/31/2026,20,22,19,21\n"
 assert builder.parse(tail,"SKEW","2026-01-01","2027-01-01")=={"2026-07-31":"145.200000"}
 assert builder.parse(ohlc,"VIX","2026-01-01","2027-01-01")=={"2026-07-31":"21.000000"}
def test_deterministic_gzip_bytes():
 assert builder.gzip_bytes(b"x\n")==builder.gzip_bytes(b"x\n")
