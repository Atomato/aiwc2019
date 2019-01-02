#!/usr/bin/python3

# keunhyung 1/2
# Test moving agent.
# Test top-20 moving models.
# Test on rule-based-postion.

from __future__ import print_function

from twisted.internet import reactor
from twisted.internet.defer import inlineCallbacks

from autobahn.wamp.serializer import MsgPackSerializer
from autobahn.wamp.types import ComponentConfig
from autobahn.twisted.wamp import ApplicationSession, ApplicationRunner

import argparse
import random
import math
import os
import sys

import base64
import numpy as np
import time

import helper

import tensorflow as tf

from args import Argument
import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers

from discrete import Discrete

#reset_reason
NONE = 0
GAME_START = 1
SCORE_MYTEAM = 2
SCORE_OPPONENT = 3
GAME_END = 4
DEADLOCK = 5

#coordinates
MY_TEAM = 0
OP_TEAM = 1
BALL = 2
X = 0
Y = 1
TH = 2
ACTIVE = 3
TOUCH = 4

class Frame(object):
    def __init__(self):
        self.time = None
        self.score = None
        self.reset_reason = None
        self.coordinates = None

class Component(ApplicationSession):

    def __init__(self, config):
        ApplicationSession.__init__(self, config)

    def printConsole(self, message):
        print(message)
        sys.__stdout__.flush()

    def onConnect(self):
        self.join(self.config.realm)

    @inlineCallbacks
    def onJoin(self, details):

##############################################################################
        def init_variables(self, info):
            # Here you have the information of the game (virtual init() in random_walk.cpp)
            # List: game_time, goal, number_of_robots, penalty_area, codewords,
            #       robot_height, robot_radius, max_linear_velocity, field, team_info,
            #       {rating, name}, axle_length, resolution, ball_radius
            # self.game_time = info['game_time']
            self.field = info['field']
            self.robot_size = 2*info['robot_radius']
            self.goal = info['goal']
            self.max_linear_velocity = info['max_linear_velocity']
            self.number_of_robots = info['number_of_robots']
            self.end_of_frame = False

            ##################################################################
            # team info, 5 robots, (x,y,th,active,touch)
            self.cur_my = [[] for _ in range(self.number_of_robots)]

            self.cur_ball = [] # ball (x,y) position
            self.prev_ball = [0., 0.] # previous ball (x,y) position

            # distance to the ball
            self.dist_ball = np.zeros(self.number_of_robots)
            # index for which robot is close to the ball
            self.idxs = [i for i in range(self.number_of_robots)]

            self.dlck_cnt = 0 # deadlock count
            # how many times avoid deadlock function was called
            self.avoid_dlck_cnt = 0

            self.wheels = np.zeros(self.number_of_robots*2)    
            ##################################################################
            self.state_dim = 2 # relative ball
            self.history_size = 2 # frame history size
            self.action_dim = 2 # 2                    
            
            # Histories of five robots.
            self.state = [np.zeros([self.state_dim * self.history_size]) \
                                    for _ in range(self.number_of_robots)]

            self.arglist = Argument()

            # state dimension
            self.state_shape = (self.state_dim * self.history_size,) 
            self.act_space = [Discrete(self.action_dim * 2 + 1)]
            self.trainers = MADDPGAgentTrainer('agent_moving', self.mlp_model, 
                                            self.state_shape, self.act_space, 
                                            0, self.arglist, local_q_func=False)
            ##################################################################
            self.load_step_list = np.loadtxt('./test_step_list.txt')
            self.step_idx = 0 # For self.load_step_list
        
            # # Load previous results.
            if self.arglist.restore:
                self.printConsole('Loading previous state... %d' % \
                                                self.load_step_list[self.step_idx])
                U.load_state('./save_model/aiwc_maddpg-%d' % \
                                                self.load_step_list[self.step_idx])
            ##################################################################
            # for tensorboard
            self.summary_placeholders, self.update_ops, self.summary_op = \
                                                            self.setup_summary()
            self.summary_writer = \
                tf.summary.FileWriter('summary/moving_test', U.get_session().graph)
            ##################################################################
            self.test_step = 0
            self.stats_steps = 12000 # For tensorboard, about 10 minutes

            self.scr_my = 0. # my team score
            self.scr_op = 0. # op team score
            self.scr_sum = 0 # score sum

            self.reset = False
            ##################################################################
            self.cur_time = time.time() # For check time to take
            return
##############################################################################        
        try:
            info = yield self.call(u'aiwc.get_info', args.key)
        except Exception as e:
            self.printConsole("Error: {}".format(e))
        else:
            try:
                self.sub = yield self.subscribe(self.on_event, args.key)
            except Exception as e2:
                self.printConsole("Error: {}".format(e2))
               
        init_variables(self, info)
        
        try:
            yield self.call(u'aiwc.ready', args.key)
        except Exception as e:
            self.printConsole("Error: {}".format(e))
        else:
            self.printConsole("I am ready for the game!")

    def mlp_model(self, input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
        # This model takes as input an observation and returns values of all actions
        with tf.variable_scope(scope, reuse=reuse):
            out = input
            out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
            return out

    # make summary operators for tensorboard
    def setup_summary(self):
        score_ratio = tf.Variable(0.) # my score / op score
        score_sum = tf.Variable(0.)

        tf.summary.scalar('Score Ratio', score_ratio)
        tf.summary.scalar('Score Sum', score_sum)

        summary_vars = [score_ratio, score_sum]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in
                                range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in
                      range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op
##############################################################################    
    def get_coord(self, received_frame):
        self.cur_ball = received_frame.coordinates[BALL]
        self.cur_my = received_frame.coordinates[MY_TEAM]                

    def get_idxs(self):
        # sort according to distant to the ball
        for i in range(self.number_of_robots):
            self.dist_ball[i] = helper.distance(self.cur_ball[X], self.cur_my[i][X], 
                                                self.cur_ball[Y], self.cur_my[i][Y])
        idxs = sorted(range(len(self.dist_ball)), key=lambda k: self.dist_ball[k])
        return idxs  

    def count_goal_area(self):
        cnt = 0
        for i in range(self.number_of_robots):
            if (abs(self.cur_my[i][X]) > 1.6) and (abs(self.cur_my[i][Y]) < 0.43):
                cnt += 1
        return cnt

    def count_penalty_area(self):
        cnt = 0
        for i in range(self.number_of_robots):
            if (abs(self.cur_my[i][X]) > 1.3) and (abs(self.cur_my[i][Y]) < 0.7):
                cnt += 1
        return cnt        

    def count_deadlock(self):
        # delta of ball
        delta_b = helper.distance(self.cur_ball[X], self.prev_ball[X], \
                                    self.cur_ball[Y], self.prev_ball[Y])

        if (abs(self.cur_ball[Y]) > 0.65) and (delta_b < 0.02):
            self.dlck_cnt += 1
        else:
            self.dlck_cnt = 0
            self.avoid_dlck_cnt = 0

    def pre_processing(self, i, x, y):
        relative_pos = helper.rot_transform(self.cur_my[i][X], 
                self.cur_my[i][Y], -self.cur_my[i][TH], x, y)

        dist = helper.distance(relative_pos[X],0,relative_pos[Y],0)
        # If distance to the ball is over 0.5,
        # clip the distance to be 0.5
        if dist > 0.5:
            relative_pos[X] *= 0.5/dist
            relative_pos[Y] *= 0.5/dist

        # Finally nomalize the distance for the maximum to be 1
        relative_pos[X] *= 2
        relative_pos[Y] *= 2

        return np.array(relative_pos)
##############################################################################
    # RL-based moving
    def position(self, i, x, y):
        relative_pos = self.pre_processing(i, x, y)
        if self.reset:
            # 2 frames position stack
            self.state[i] = np.append(relative_pos, relative_pos)
        else:
            # 2 frames position stack
            self.state[i] = np.append(relative_pos, self.state[i][:-self.state_dim])

        self.printConsole(self.state[i])

        # get action
        self.action = self.trainers.action(self.state[i])

        self.wheels[2*i] = self.max_linear_velocity * \
                (self.action[1]-self.action[2]+self.action[3]-self.action[4])
        self.wheels[2*i + 1] = self.max_linear_velocity * \
                (self.action[1]-self.action[2]-self.action[3]+self.action[4])
##############################################################################
    @inlineCallbacks
    def on_event(self, f):        

        @inlineCallbacks
        def set_wheel(self, robot_wheels):
            yield self.call(u'aiwc.set_speed', args.key, robot_wheels)
            return

        def avoid_goal_foul(self):
            midfielder(self, self.idxs[0])
            midfielder(self, self.idxs[1])
            self.position(self.idxs[2], 0, 0)
            self.position(self.idxs[3], 0, 0)
            self.position(self.idxs[4], 0, 0)

        def avoid_penalty_foul(self):
            midfielder(self, self.idxs[0])
            midfielder(self, self.idxs[1])
            midfielder(self, self.idxs[2])
            self.position(self.idxs[3], 0, 0)
            self.position(self.idxs[4], 0, 0)            

        def avoid_deadlock(self):
            self.position(0, self.cur_ball[X], 0)
            self.position(1, self.cur_ball[X], 0)
            self.position(2, self.cur_ball[X], 0)
            self.position(3, self.cur_ball[X], 0)
            self.position(4, self.cur_ball[X], 0)

            # if closest ball is somhow away from the ball
            # or avoided deadlock to some extent
            if (self.dist_ball[self.idxs[0]] > 0.13) or (self.avoid_dlck_cnt > 20):
                offense(self)            

        def midfielder(self, robot_id):
            goal_dist = helper.distance(self.cur_my[robot_id][X], self.field[X]/2,
                                         self.cur_my[robot_id][Y], 0)
            shoot_mul = 1
            dribble_dist = 0.426
            v = 5
            goal_to_ball_unit = helper.unit([self.field[X]/2 - self.cur_ball[X],
                                                            - self.cur_ball[Y]])
            delta = [self.cur_ball[X] - self.cur_my[robot_id][X],
                    self.cur_ball[Y] - self.cur_my[robot_id][Y]]

            if (self.dist_ball[robot_id] < 0.5) and (delta[X] > 0):
                self.position(robot_id, self.cur_ball[X] + v*delta[X], 
                                        self.cur_ball[Y] + v*delta[Y])                   
            else:
                self.position(robot_id, 
                    self.cur_ball[X] - dribble_dist*goal_to_ball_unit[X], 
                    self.cur_ball[Y] - dribble_dist*goal_to_ball_unit[Y])

        def offense(self):
            midfielder(self, 0)
            midfielder(self, 1)
            midfielder(self, 2)
            midfielder(self, 3)
            midfielder(self, 4)

        def set_formation(self):
            # count how many robots is in the goal area
            goal_area_cnt = self.count_goal_area()
            # count how many robots is in the penalty area
            penalty_area_cnt = self.count_penalty_area()
            self.count_deadlock()

            if goal_area_cnt > 2:
                avoid_goal_foul(self)
                self.printConsole('avoid goal foul')
            elif penalty_area_cnt > 3:
                avoid_penalty_foul(self)
                self.printConsole('avoid penalty foul')
            elif self.dlck_cnt > 15:
                avoid_deadlock(self)
                self.printConsole('avoid deadlock')
                self.avoid_dlck_cnt += 1
            else:
                offense(self)
                self.printConsole('offense')
            
        # initiate empty frame
        received_frame = Frame()

        if 'time' in f:
            received_frame.time = f['time']
        if 'score' in f:
            received_frame.score = f['score']
        if 'reset_reason' in f:
            received_frame.reset_reason = f['reset_reason']
        if 'coordinates' in f:
            received_frame.coordinates = f['coordinates']            
        if 'EOF' in f:
            self.end_of_frame = f['EOF']
        
        #self.printConsole(received_frame.time)
        #self.printConsole(received_frame.score)
        #self.printConsole(received_frame.reset_reason)
        #self.printConsole(self.end_of_frame)
##############################################################################
        if (self.end_of_frame):
            
            # How to get the robot and ball coordinates: (ROBOT_ID can be 0,1,2,3,4)
            #self.printConsole(received_frame.coordinates[MY_TEAM][ROBOT_ID][X])            
            #self.printConsole(received_frame.coordinates[MY_TEAM][ROBOT_ID][Y])
            #self.printConsole(received_frame.coordinates[MY_TEAM][ROBOT_ID][TH])
            #self.printConsole(received_frame.coordinates[MY_TEAM][ROBOT_ID][ACTIVE])
            #self.printConsole(received_frame.coordinates[MY_TEAM][ROBOT_ID][TOUCH])
            #self.printConsole(received_frame.coordinates[OP_TEAM][ROBOT_ID][X])
            #self.printConsole(received_frame.coordinates[OP_TEAM][ROBOT_ID][Y])
            #self.printConsole(received_frame.coordinates[OP_TEAM][ROBOT_ID][TH])
            #self.printConsole(received_frame.coordinates[OP_TEAM][ROBOT_ID][ACTIVE])
            #self.printConsole(received_frame.coordinates[OP_TEAM][ROBOT_ID][TOUCH])
            #self.printConsole(received_frame.coordinates[BALL][X])
            #self.printConsole(received_frame.coordinates[BALL][Y])
##############################################################################
            self.get_coord(received_frame)
            self.idxs  = self.get_idxs()

            # Reset
            if (received_frame.reset_reason == SCORE_MYTEAM):
                self.reset = True
                self.scr_my += 1
                self.scr_sum += 1
                self.printConsole("reset reason: " + \
                    str(received_frame.reset_reason))                
            elif (received_frame.reset_reason == SCORE_OPPONENT):
                self.reset = True
                self.scr_op += 1
                self.scr_sum -= 1
                self.printConsole("reset reason: " + \
                    str(received_frame.reset_reason))                
            elif(received_frame.reset_reason != NONE) or \
                                    (received_frame.reset_reason == None):
                self.reset = True
                self.printConsole("reset reason: " + \
                    str(received_frame.reset_reason))
            else:
                self.reset = False

            set_formation(self) # rule-based formation
            set_wheel(self, self.wheels.tolist())
            self.prev_ball = self.cur_ball            


            # increment global step counter
            self.test_step += 1
            self.printConsole('step: ' + str(self.test_step))
            if (self.test_step % 1200) == 0:
                self.printConsole('%d seconds' %(time.time()-self.cur_time))
                self.cur_time = time.time()
##############################################################################
            # plot every 72000 steps (about 10 minutes)
            if ((self.test_step % self.stats_steps) == 0) and (self.step_idx < 20):
                score_ratio = self.scr_my / self.scr_op \
                                if self.scr_op != 0. else 100

                stats = [score_ratio, self.scr_sum]
                for i in range(len(stats)):
                    U.get_session().run(self.update_ops[i], feed_dict={
                        self.summary_placeholders[i]: float(stats[i])
                    })
                summary_str = U.get_session().run(self.summary_op)
                self.summary_writer.add_summary(summary_str, 
                    self.load_step_list[self.step_idx])

                self.step_idx += 1
                self.scr_my, self.scr_op, self.scr_sum = 0,0,0

                # load new model
                print('Loading %s' % self.load_step_list[self.step_idx])
                U.load_state('./save_model/aiwc_maddpg-%d' % \
                                self.load_step_list[self.step_idx])
##############################################################################
            if(received_frame.reset_reason == GAME_END):
                #(virtual finish() in random_walk.cpp)
                #save your data
                with open(args.datapath + '/result.txt', 'w') as output:
                    #output.write('yourvariables')
                    output.close()
                #unsubscribe; reset or leave  
                yield self.sub.unsubscribe()
                try:
                    yield self.leave()
                except Exception as e:
                    self.printConsole("Error: {}".format(e))

            self.end_of_frame = False
##############################################################################

    def onDisconnect(self):
        if reactor.running:
            reactor.stop()

if __name__ == '__main__':
    
    try:
        unicode
    except NameError:
        # Define 'unicode' for Python 3
        def unicode(s, *_):
            return s

    def to_unicode(s):
        return unicode(s, "utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("server_ip", type=to_unicode)
    parser.add_argument("port", type=to_unicode)
    parser.add_argument("realm", type=to_unicode)
    parser.add_argument("key", type=to_unicode)
    parser.add_argument("datapath", type=to_unicode)
    
    args = parser.parse_args()
    
    ai_sv = "rs://" + args.server_ip + ":" + args.port
    ai_realm = args.realm
    
    with U.single_threaded_session():
        # create a Wamp session object
        session = Component(ComponentConfig(ai_realm, {}))

        # initialize the msgpack serializer
        serializer = MsgPackSerializer()
        
        # use Wamp-over-rawsocket
        runner = ApplicationRunner(ai_sv, ai_realm, serializers=[serializer])
        
        runner.run(session, auto_reconnect=True)
