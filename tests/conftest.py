"""Shared realistic VersaStudio text fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture
def par_bytes() -> bytes:
    return b"""VersaStudio File
Cycles=2
Scan Rate (V/s)=0.05
Segments=4
Vertex 1 (V)=0.50
Vertex 2 (V)=-0.20
<Segment1>
Type=2
Version=3
Definition=Segment #, Point #, E(V), I(A), Elapsed Time(s), E Applied(V), Status
1,0,-0.20,-1.0E-06,0.0,-0.20,0
1,1,0.00,1.0E-06,1.0,0.00,0
1,2,0.20,5.0E-06,2.0,0.20,0
1,3,0.40,2.0E-06,3.0,0.40,0
</Segment1>
<Segment2>
Type=2
Version=3
Definition=Segment #, Point #, E(V), I(A), Elapsed Time(s), E Applied(V), Status
2,0,0.40,2.0E-06,3.1,0.40,0
2,1,0.20,-1.0E-06,4.0,0.20,0
broken,row
2,2,0.00,-4.0E-06,5.0,0.00,0
2,3,-0.20,-1.0E-06,6.0,-0.20,0
</Segment2>
<Segment3>
Definition=Segment #, Point #, E(V), I(A), Elapsed Time(s), E Applied(V), Status
3,0,-0.20,-1.1E-06,7.0,-0.20,0
3,1,0.00,1.2E-06,8.0,0.00,0
3,2,0.20,5.5E-06,9.0,0.20,0
3,3,0.40,2.2E-06,10.0,0.40,0
</Segment3>
<Segment4>
Definition=Segment #, Point #, E(V), I(A), Elapsed Time(s), E Applied(V), Status
4,0,0.40,2.2E-06,10.1,0.40,0
4,1,0.20,-1.2E-06,11.0,0.20,0
4,2,0.00,-4.5E-06,12.0,0.00,0
4,3,-0.20,-1.1E-06,13.0,-0.20,0
</Segment4>
"""
