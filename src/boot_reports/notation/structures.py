

types = ['layout',
         'csensor',
         'dsensor',
         'mounted-sensor',
         'joint',
         'number',
         'dynamics',
         'param']


def add(name, op, types, result):
    pass


#def addprim(name, op, type):
#    pass

add('semi-circle', 'c', 'layout')
add('fullcircle', 'o', 'layout')
add('line', 'l', 'layout')
add('square', 's', 'layout')

add('sampling', 'S', ['param', 'csensor'], 'csensor')

add('range-finder', 'r', ['layout'], 'csensor')
add('vision', 'v', ['layout'], 'csensor')
add('field-sampler', 'fs', ['layout'], 'csensor')

add('smoothing', 'S', ['param', 'csensor'], 'csensor')

add('make joint', 'j', ['number'], 'mounted-sensor')
add('mount on joint', 'M', ['joint', 'dsensor'], 'mounted-sensor')
