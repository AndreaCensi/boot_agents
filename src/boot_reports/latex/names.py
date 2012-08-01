

def bvid(id_component):
    return "\\bvid{%s}" % id_component


def bvrobpretty(id_robot):
    """ Pretty prints robot ID as formula """
    return "S(e, %s)" % bvid(id_robot) # FIXME
