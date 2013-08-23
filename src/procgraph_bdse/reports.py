from reprep import Report
from procgraph import simple_block
from reprep.structures import NotExistent
from compmake.utils.describe import describe_type
from reprep.datanode import DataNode

@simple_block
def make_agent_report(agent, subneeded=None):
    r = Report()
    
    r.set_subsections_needed(subneeded)
    
    agent.publish(r)
    return r

from procgraph import Block

class ImgFromReport(Block):
    Block.alias('img_from_report')
    Block.config('url', 'subset of report')
    Block.input('report')
    Block.output('img')
    
    def init(self):
        pass
    
    def update(self):
        r = self.input.report
#         self.info(r.format_tree())
        
        url = self.config.url
        try:
            node = r.resolve_url(url)
        except NotExistent as e:
            # print r['warn'].raw_data
            self.debug(e) 
            return
        if not isinstance(node, DataNode):
            msg = 'Expected datanode, got %r' % describe_type(node)
            raise Exception(msg)
        self.output.img = node.raw_data
            
    
        
