#!/usr/bin/python3

# keunhyung 12/27
# shoot and chase
# multi experience

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
            self.cur_my_posture = []
            self.cur_ball = []

            self.state_dim = 2 # relative ball
            self.history_size = 2 # frame history size
            self.action_dim = 2 # 2                    
            
            self.arglist = Argument()
            self.state_shape = (self.state_dim * self.history_size,) # state dimension
            self.act_space = [Discrete(self.action_dim * 2 + 1)]
            self.trainers = MADDPGAgentTrainer(
                'agent_moving', self.mlp_model, self.state_shape, self.act_space, 0, self.arglist,
                local_q_func=False)

            # for tensorboard
            # self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summary()
            # self.summary_writer = tf.summary.FileWriter('summary/aiwc_maddpg', U.get_session().graph)

            U.initialize()
            
            # Load previous results, if necessary
            if self.arglist.load_dir == "":
                self.arglist.load_dir = self.arglist.save_dir
            if self.arglist.restore:
                print('Loading previous state...')
                U.load_state(self.arglist.load_dir)

            self.episode_rewards = [0.0]  # sum of rewards for all agents
            self.agent_rewards = [[0.0] for _ in range(self.number_of_robots)]  # individual agent reward

            self.saver = tf.train.Saver()

            self.state = np.zeros([self.state_dim * self.history_size]) # histories
            self.train_step = 0
            self.wheels = np.zeros(self.number_of_robots*2)
            self.action = np.zeros(self.action_dim * 2 + 1) # not np.zeros(2)
                   
            self.save_every_steps = 12000 # save the model every 10 minutes
            self.stats_steps = 6000 # for tensorboard
            self.reward_sum = np.zeros(self.number_of_robots)
            self.score_sum = 0 
            self.inner_step = 0

            self.done = False
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
        episode_0_reward = tf.Variable(0.)
        episode_1_reward = tf.Variable(0.)
        episode_2_reward = tf.Variable(0.)
        episode_3_reward = tf.Variable(0.)
        episode_4_reward = tf.Variable(0.)
        episode_total_score = tf.Variable(0.)

        tf.summary.scalar('Total Reward/Episode', episode_total_reward)
        tf.summary.scalar('Reward 0/Episode', episode_0_reward)
        tf.summary.scalar('Reward 1/Episode', episode_1_reward)
        tf.summary.scalar('Reward 2/Episode', episode_2_reward)
        tf.summary.scalar('Reward 3/Episode', episode_3_reward)
        tf.summary.scalar('Reward 4/Episode', episode_4_reward)
        tf.summary.scalar('Total Score/Episode', episode_total_score)

        summary_vars = [episode_total_reward, episode_0_reward, episode_1_reward, 
            episode_2_reward, episode_3_reward, episode_4_reward, episode_total_score]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in
                                range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in
                      range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op
##############################################################################    
    def get_coord(self, received_frame):
        self.cur_ball = received_frame.coordinates[BALL]
        self.cur_my_posture = received_frame.coordinates[MY_TEAM]                

    def get_reward(self, reset_reason, i):
        dist_rew = -0.1*helper.distance(self.cur_ball[X], self.cur_my_posture[i][X], 
            self.cur_ball[Y], self.cur_my_posture[i][Y])
        self.printConsole('         distance reward ' + str(i) + ': ' + str(dist_rew))

        touch_rew = 0
        if self.cur_my_posture[i][TOUCH]:
            touch_rew += 10

        rew = dist_rew + touch_rew

        self.printConsole('                 reward ' + str(i) + ': ' + str(rew))
        return rew      

    def pre_processing(self, i):
        relative_ball = helper.rot_transform(self.cur_my_posture[i][X], 
            self.cur_my_posture[i][Y], -self.cur_my_posture[i][TH], 
            self.cur_ball[X], self.cur_ball[Y])

        self.printConsole('Original input: %s' % relative_ball)

        dist = helper.distance(relative_ball[X],0,relative_ball[Y],0)
        # If distance to the ball is over 0.5,
        # clip the distance to be 0.5
        if dist > 0.5:
            relative_ball[X] *= 0.5/dist
            relative_ball[Y] *= 0.5/dist

        # Finally nomalize the distance for the maximum to be 1
        relative_ball[X] *= 2
        relative_ball[Y] *= 2

        self.printConsole('Final input: %s' % relative_ball)

        return np.array(relative_ball)
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
        
        dx = x - self.cur_my_posture[robot_id][X]
        dy = y - self.cur_my_posture[robot_id][Y]
        d_e = math.sqrt(math.pow(dx, 2) + math.pow(dy, 2))
        desired_th = (math.pi/2) if (dx == 0 and dy == 0) else math.atan2(dy, dx)

        d_th = desired_th - self.cur_my_posture[robot_id][TH] 
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
                                               
            self.get_coord(received_frame)
##############################################################################
            # Next state, Reward, Reset
            if self.done:
                self.control_idx += 1
                self.control_idx %= 5

            # Next state
            next_obs = self.pre_processing(self.control_idx)
            if self.done:
                next_state = np.append(next_obs, next_obs) # 2 frames position stack
                self.done = False
            else:
                next_state = np.append(next_obs, self.state[:-self.state_dim]) # 2 frames position stack

            # Reward
            reward = self.get_reward(received_frame.reset_reason, self.control_idx)

            # Reset
            if(received_frame.reset_reason != NONE) and (received_frame.reset_reason is not None):
                self.done = True
                self.printConsole("reset reason: " + str(received_frame.reset_reason))
            else:
                self.done = False

            if not self.cur_my_posture[self.control_idx][ACTIVE]:
                self.printConsole('robot ' + str(self.control_idx) + ' is not active')
            else:
                self.trainers.experience(self.state, self.action, reward, next_state, self.done, False)

            self.state = next_state

            # increment global step counter
            self.train_step += 1

            # update 
            self.trainers.preupdate()
            loss = self.trainers.update([self.trainers], self.train_step)

            # get action
            self.action = self.trainers.action(self.state)

            self.wheels = np.zeros(self.number_of_robots*2)
            self.wheels[2*self.control_idx] = self.max_linear_velocity * \
                    (self.action[1]-self.action[2]+self.action[3]-self.action[4])
            self.wheels[2*self.control_idx + 1] = self.max_linear_velocity * \
                    (self.action[1]-self.action[2]-self.action[3]+self.action[4])

            # Send non-control robot to the side of the field
            for i in range(self.number_of_robots):
                if i == self.control_idx:
                    continue
                else:
                    if (i == 0) or (i == 2):
                        x = self.cur_my_posture[i][X]
                        y = -1.35 
                    elif (i == 1) or (i == 3):
                        x = self.cur_my_posture[i][X]
                        y = 1.35
                    else:
                        x = -2.1
                        y = 0
                    self.position(i, x, y)

            self.printConsole('step: ' + str(self.train_step))
            # self.printConsole('wheels %s' % self.wheels)

            set_wheel(self, self.wheels.tolist())
##############################################################################
            # if (self.train_step % self.save_every_steps) == 0:
                # U.save_state(self.arglist.save_dir, saver=self.saver)

            # if done: # plot the statics
            if (self.train_step % self.stats_steps) == 0: # plot every 6000 steps (about 5 minuites)
                self.printConsole("add data to tensorboard")
                # stats = [sum(self.reward_sum)] + [self.reward_sum[i] for i in range(len(self.reward_sum))] + [self.score_sum]
                # for i in range(len(stats)):
                #     U.get_session().run(self.update_ops[i], feed_dict={
                #         self.summary_placeholders[i]: float(stats[i])
                #     })
                # summary_str = U.get_session().run(self.summary_op)
                # self.summary_writer.add_summary(summary_str, self.inner_step)

                # self.reward_sum = np.zeros(len(self.reward_sum))
                self.score_sum = 0
                self.inner_step += 1            
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
