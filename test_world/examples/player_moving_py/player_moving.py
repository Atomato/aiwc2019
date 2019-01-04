#!/usr/bin/python3

# keunhyung 1/4
# Learn how to move
# multi experience
# quarter state aggregation
# reward: theta and ball touch

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

            self.wheels = np.zeros(self.number_of_robots*2)
            ##################################################################
            self.state_dim = 2 # relative ball
            self.history_size = 2 # frame history size
            self.action_dim = 2 + 1 # (stay, straight, rotate)

            self.state = np.zeros([self.state_dim * self.history_size]) # histories
            self.action = np.zeros(self.action_dim)
            
            self.arglist = Argument()

            self.state_shape = (self.state_dim * self.history_size,)
            self.act_space = [Discrete(self.action_dim)]
            self.trainers = MADDPGAgentTrainer(
                'agent_moving', self.mlp_model, self.state_shape, self.act_space, 0, self.arglist,
                local_q_func=False)
            ##################################################################
            # for tensorboard
            self.summary_placeholders, self.update_ops, self.summary_op = \
                                                            self.setup_summary()
            self.summary_writer = \
                tf.summary.FileWriter('summary/moving_quarter_2', U.get_session().graph)

            U.initialize()
            ##################################################################
            # Load previous results, if necessary
            if self.arglist.load_dir == "":
                self.arglist.load_dir = self.arglist.save_dir
            if self.arglist.restore:
                print('Loading previous state... %s' % self.arglist.load_dir)
                U.load_state(self.arglist.load_dir)

            self.saver = tf.train.Saver(max_to_keep=1100)
            ##################################################################
            self.train_step = 0
                   
            self.save_every_steps = 6000 # save the model every 5 minutes
            self.stats_steps = 6000 # for tensorboard
            self.rwd_sum = 0

            self.reset = False
            self.control_idx = 0
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
        episode_total_reward = tf.Variable(0.)

        tf.summary.scalar('Total Reward/Episode', episode_total_reward)

        summary_vars = [episode_total_reward]
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

    def get_reward(self, i, th):
        theta_rwd = -th

        touch_rwd = 0
        if self.cur_my[i][TOUCH]:
            touch_rwd += 1

        rwd = theta_rwd + touch_rwd
        self.printConsole('Reward %s' [theta_rwd, touch_rwd, rwd])
        return rwd

    def pre_processing(self, i):
        # Radius and theta.
        r, th = helper.polar_transform(self.cur_my[i][X], 
                    self.cur_my[i][Y], -self.cur_my[i][TH], 
                    self.cur_ball[X], self.cur_ball[Y])

        self.printConsole('Original position %d: %s' % (i, [r, th]))

        # If distance to the ball is over 0.5,
        # clip the distance to be 0.5
        if r > 0.5:
            r = 0.5
        # Finally nomalize the distance for the maximum to be 1.
        r *= 2

        if (0<=th) and (th<90):
            th /= 90 # Normalize.
            quadrant = 1
        elif (90<=th) and (th<=180):
            th = -th + 180 # Make theta to be 0 to 90.
            th /= 90 # Normalize.
            quadrant = 2
        elif (-180<=th) and (th<-90):
            th = th + 180 # Make theta to be 0 to 90.
            th /= 90 # Normalize.
            quadrant = 3
        elif (-90<=th) and (th<0):
            th = -th # Make theta to be 0 to 90.
            th /= 90 # Normalize.
            quadrant = 4
        else:
            th = 0
            print('Theta error')

        self.printConsole('Final position %d: %s' % (i, [r, th, quadrant]))

        return np.array([r, th]), quadrant

    def step(self, received_frame):
        # Reset
        if(received_frame.reset_reason != NONE) or \
            (received_frame.reset_reason == None):
            self.reset = True
            self.printConsole("Reset reason: " + str(received_frame.reset_reason))
        else:
            self.reset = False

        # Next state
        next_obs, quadrant = self.pre_processing(self.control_idx)
        if self.reset:
            next_state = np.append(next_obs, next_obs) # 2 frames position stack
            self.reset = False
        else:
            # 2 frames position stack
            next_state = np.append(next_obs, self.state[:-self.state_dim])
        self.printConsole('Next state: %s' % next_state)

        # Reward
        reward = self.get_reward(self.control_idx, next_obs[1])

        return reward, next_state, reset, quadrant
##############################################################################
    def action_to_wheel(self, action, quadrant):
        if quadrant == 1:
            left_wheel = self.max_linear_velocity * (action[1]-action[2])
            right_wheel = self.max_linear_velocity * (action[1]+action[2])
        elif quadrant == 2:
            left_wheel = self.max_linear_velocity * (-action[1]+action[2])
            right_wheel = self.max_linear_velocity * (-action[1]-action[2])
        elif quadrant == 3:
            left_wheel = self.max_linear_velocity * (-action[1]-action[2])
            right_wheel = self.max_linear_velocity * (-action[1]+action[2])
        elif quadrant == 4:
            left_wheel = self.max_linear_velocity * (action[1]+action[2])
            right_wheel = self.max_linear_velocity * (action[1]-action[2])
        else:
            self.printConsole('quadrant error')
        self.printConsole('Wheels: %s' % [left_wheel, right_wheel])

        return left_wheel, right_wheel
##############################################################################
    # function for heuristic moving
    def set_wheel_velocity(self, robot_id, left_wheel, right_wheel):
        multiplier = 1
        
        if(abs(left_wheel) > self.max_linear_velocity or abs(right_wheel) > self.max_linear_velocity):
            if (abs(left_wheel) > abs(right_wheel)):
                multiplier = self.max_linear_velocity / abs(left_wheel)
            else:
                multiplier = self.max_linear_velocity / abs(right_wheel)
        
        self.wheels[2*robot_id] = left_wheel*multiplier
        self.wheels[2*robot_id + 1] = right_wheel*multiplier

    def position(self, robot_id, x, y):
        damping = 0.35
        mult_lin = 3.5
        mult_ang = 0.4
        ka = 0
        sign = 1
        
        dx = x - self.cur_my[robot_id][X]
        dy = y - self.cur_my[robot_id][Y]
        d_e = math.sqrt(math.pow(dx, 2) + math.pow(dy, 2))
        desired_th = (math.pi/2) if (dx == 0 and dy == 0) else math.atan2(dy, dx)

        d_th = desired_th - self.cur_my[robot_id][TH] 
        while(d_th > math.pi):
            d_th -= 2*math.pi
        while(d_th < -math.pi):
            d_th += 2*math.pi
            
        if (d_e > 1):
            ka = 17/90
        elif (d_e > 0.5):
            ka = 19/90
        elif (d_e > 0.3):
            ka = 21/90
        elif (d_e > 0.2):
            ka = 23/90
        else:
            ka = 25/90
            
        if (d_th > helper.degree2radian(95)):
            d_th -= math.pi
            sign = -1
        elif (d_th < helper.degree2radian(-95)):
            d_th += math.pi
            sign = -1
            
        if (abs(d_th) > helper.degree2radian(85)):
            self.set_wheel_velocity(robot_id, -mult_ang*d_th, mult_ang*d_th)
        else:
            if (d_e < 5 and abs(d_th) < helper.degree2radian(40)):
                ka = 0.1
            ka *= 4
            self.set_wheel_velocity(robot_id, 
                                    sign * (mult_lin * (1 / (1 + math.exp(-3*d_e)) - damping) - mult_ang * ka * d_th),
                                    sign * (mult_lin * (1 / (1 + math.exp(-3*d_e)) - damping) + mult_ang * ka * d_th))
##############################################################################

    @inlineCallbacks
    def on_event(self, f):        

        @inlineCallbacks
        def set_wheel(self, robot_wheels):
            yield self.call(u'aiwc.set_speed', args.key, robot_wheels)
            return
            
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
            # Train only when the robot is active.
            if self.cur_my[self.control_idx][ACTIVE]:
                # Next state, Reward, Reset
                # if self.reset:
                #     self.control_idx += 1
                #     self.control_idx %= 5

                reward, next_state, reset, quadrant = self.step(received_frame)

                self.trainers.experience(self.state, self.action, reward, 
                    next_state, self.reset, False)

                # update 
                self.trainers.preupdate()
                loss = self.trainers.update([self.trainers], self.train_step)

                self.state = next_state
                # get action
                self.action = self.trainers.action(self.state)
                self.printConsole('Original action %s' % self.action)

                left_wheel, right_wheel = self.action_to_wheel(
                                            self.action, quadrant)

                self.wheels[2*self.control_idx] = left_wheel
                self.wheels[2*self.control_idx + 1] = right_wheel

                # Send non-control robot to the side of the field
                for i in range(self.number_of_robots):
                    if i == self.control_idx:
                        continue
                    else:
                        if (i == 0) or (i == 2):
                            x = self.cur_my[i][X]
                            y = -1.35 
                        elif (i == 1) or (i == 3):
                            x = self.cur_my[i][X]
                            y = 1.35
                        else:
                            x = -2.1
                            y = 0
                        self.position(i, x, y)

                set_wheel(self, self.wheels.tolist())

                # Increase global step counter.
                # Increase count only when the control robot is active.
                self.train_step += 1
                self.rwd_sum += reward
                self.printConsole('step: %d\n' % self.train_step)
            else:
                self.printConsole('The control robot is not active.')
##############################################################################
            if (self.train_step % self.save_every_steps) == 0:
                self.saver.save(U.get_session(), self.arglist.save_dir, 
                    global_step = self.train_step)

            # plot every 6000 steps (about 5 minuites)
            if (self.train_step % self.stats_steps) == 0:
                stats = [self.rwd_sum]
                for i in range(len(stats)):
                    U.get_session().run(self.update_ops[i], feed_dict={
                        self.summary_placeholders[i]: float(stats[i])})
                summary_str = U.get_session().run(self.summary_op)
                self.summary_writer.add_summary(summary_str, self.train_step)

                self.rwd_sum = 0         
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
