from .sync_box import get_sync_deriv_box
from blocks.library import SampledDerivInst
from blocks.unittests import BlocksTest


class SyncBoxTest(BlocksTest):
    
    def deriv_inst_test(self):
        # This is normal behavior for SampledDerivInst(
        data = [
            (0.0, 10.0),
            (1.0, 11.0),
            (2.0, 12.0),
            (3.0, 13.0),
            (4.0, 14.0),
            (5.0, 14.0),
            (6.0, 14.0),
        ]
        expected = [
            (0.0, (10.0, +0.0)),
            (1.0, (11.0, +1.0)),
            (2.0, (12.0, +1.0)),
            (3.0, (13.0, +1.0)),
            #(4.0, (14.0, +0.5)), # other behavior
            (4.0, (14.0, +1.0)),
            (5.0, (14.0, +0.0)),
            (6.0, (14.0, +0.0)),
        ]
        bbox = SampledDerivInst()
        self.check_bbox_results(bbox, data, expected)
 
 
    def sync_box_test1(self):
         
        data = [
            (0.0, ('u', 3.0)),
            (0.0, ('y', 10.0)),
            (0.5, ('u', 4.0)),
            (1.0, ('y', 11.0)),
            (1.5, ('u', 5.0)),
            (1.6, ('u', 6.0)),
            (2.0, ('y', 12.0)),
            (2.5, ('u', 6.0)),
            (3.0, ('y', 13.0)),
            (3.5, ('u', 7.0)),
            (4.0, ('y', 14.0)),
            (4.5, ('u', 6.0)),
            (5.0, ('y', 14.0)),
            (5.5, ('u', 6.0)),
            (6.0, ('y', 14.0)),
        ]
        
        expected = [
            (0.0, ('y_u', dict(y=10.0, y_dot=+0.0, u=3.0))),
            (1.0, ('y_u', dict(y=11.0, y_dot=+1.0, u=4.0))),
            (2.0, ('y_u', dict(y=12.0, y_dot=+1.0, u=6.0))),
            (3.0, ('y_u', dict(y=13.0, y_dot=+1.0, u=6.0))),
            (4.0, ('y_u', dict(y=14.0, y_dot=+1.0, u=7.0))),
            (5.0, ('y_u', dict(y=14.0, y_dot=+0.0, u=6.0))),
            (6.0, ('y_u', dict(y=14.0, y_dot=+0.0, u=6.0))),
        ]
        
        bbox = get_sync_deriv_box()

        self.check_bbox_results(bbox, data, expected)
