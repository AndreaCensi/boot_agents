from procgraph import  Block
import Image
from procgraph_ros.conversions import pil_to_imgmsg
import numpy as np
from procgraph_images.copied_from_reprep import posneg

class ToDPLog(Block):
    ''' Same filtering as PureCommands '''

    Block.alias('to_dp_log')

    Block.config('bag', 'output bag file')

    Block.input('tuples')
    

    def init(self):
        bagfile = self.config.bag
        self.info('Writing to bag file %r.' % bagfile)

        import rosbag
        self.out_bag = rosbag.Bag(bagfile, 'w')
        
    def update(self):
        last = self.input.tuples
        timestamp = self.get_input_timestamp(0)
        from rospy import rostime
        from std_msgs.msg import Float32MultiArray
        ros_stamp = rostime.Time.from_sec(timestamp)

        y0 = last.y0
        y1 = last.y1
        u = last.commands

        cmd_msg = Float32MultiArray()
        cmd_msg.data = u.tolist()

        def image2ros(y):
            # This is a hack for now
            assert np.max(y) <= 1.0
            assert np.min(y) >= 0.0
            # convert to grayscale
            y = (y * 255.0).astype('uint8')
            msg = pil_to_imgmsg(Image.fromarray(y))
            msg.header.stamp = ros_stamp
            return msg
    
        bag = self.out_bag
        bag.write('Y0', image2ros(y0), ros_stamp)
        bag.write('Y1', image2ros(y1), ros_stamp)
        bag.write('U0', cmd_msg, ros_stamp)

        both_rgb = posneg(y0 - y1)
        both_msg = pil_to_imgmsg(Image.fromarray(both_rgb))
        both_msg.header.stamp = ros_stamp
        bag.write('both', both_msg, ros_stamp)
        
            
    def finish(self):
        self.out_bag.close()
