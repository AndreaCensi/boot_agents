from procgraph import Block
from boot_agents.diffeo.library import DiffeoLibrary
from conf_tools.exceptions import BadConfig
from procgraph.core.exceptions import BadInput
from boot_agents.diffeo.diffeo_basic import diffeo_from_function, diffeo_apply


class Diffeo(Block):
    ''' Applies a diffeomorphism. '''

    Block.alias('diffeo')

    available = DiffeoLibrary.diffeos.keys()
    Block.config('f', 'Which function to use (one of %s)'
                        % ", ".join(available))

    Block.input('rgb', 'Input image (either gray or RGB)')
    Block.output('rgb', 'Output image')

    def init(self):
        fid = self.config.f
        if not fid in DiffeoLibrary.diffeos:
            msg = 'Could not find diffeo %r.' % fid
            raise BadConfig(msg, self, 'f')
        self.f = DiffeoLibrary.diffeos[fid]
        self.D = None

    def update(self):
        rgb = self.input.rgb
        if rgb.ndim == 2:
            shape = rgb.shape
        elif rgb.ndim == 3:
            shape = (rgb.shape[0], rgb.shape[1])
        else:
            raise BadInput('expected rgb', self, 'rgb')

        if self.D is None:
            # create diffeormorphism if needed
            self.info('Creating diffeormophism %s at %s' % (self.f, shape))
            self.D = diffeo_from_function(shape, self.f)
            self.info('Done.')

        self.output.rgb = diffeo_apply(self.D, rgb)



