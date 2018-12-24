#!/usr/bin/python3

# keunhyung 12/24
# test coach using DQN
# rule-based robot players

from __future__ import print_function

from twisted.internet import reactor
from twisted.internet.defer import inlineCallbacks

from autobahn.wamp.serializer import MsgPackSerializer
from autobahn.wamp.types import ComponentConfig
from autobahn.twisted.wamp import ApplicationSession, ApplicationRunner

import argparse
import sys
import numpy as np
import math

import helper
from aiwc_dqn import DQNAgent

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
            self.cur_op = [[] for _ in range(self.number_of_robots)]

            self.cur_ball = [] # ball (x,y) position
            self.prev_ball = [0., 0.] # previous ball (x,y) position

            # distance to the ball, (team, robot index)
            self.dist_ball = np.zeros((2, self.number_of_robots))
            # index for which my robot is close to the ball
            self.idxs = [i for i in range(self.number_of_robots)]

            self.dlck_cnt = 0 # deadlock count
            # how many times avoid deadlock function was called
            self.avoid_dlck_cnt = 0             

            self.wheels = np.zeros(self.number_of_robots*2)
            ##################################################################
            self.state_dim = 3 # relative ball
            self.action_dim = 2 # avoid dead lock or not
            self.coach_agent = DQNAgent(self.state_dim, self.action_dim)
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
##############################################################################
    def get_coord(self, received_frame):
        self.cur_ball = received_frame.coordinates[BALL]
        self.cur_my = received_frame.coordinates[MY_TEAM]
        self.cur_op = received_frame.coordinates[OP_TEAM]

    def get_idxs(self):
        # sort according to distant to the ball
        # my team
        for i in range(self.number_of_robots):
            self.dist_ball[MY_TEAM][i] = helper.distance(self.cur_ball[X], 
                            self.cur_my[i][X], self.cur_ball[Y], self.cur_my[i][Y])
        # opponent team
        for i in range(self.number_of_robots):
            self.dist_ball[OP_TEAM][i] = helper.distance(self.cur_ball[X], 
                            self.cur_op[i][X], self.cur_ball[Y], self.cur_op[i][Y])

        # my team distance index
        idxs = sorted(range(len(self.dist_ball[MY_TEAM])), 
                                        key=lambda k: self.dist_ball[MY_TEAM][k])
        return idxs

    def count_deadlock(self):
        # delta of ball
        delta_b = helper.distance(self.cur_ball[X], self.prev_ball[X], \
                                    self.cur_ball[Y], self.prev_ball[Y])

        if (abs(self.cur_ball[Y]) > 0.65) and (delta_b < 0.02):
            self.dlck_cnt += 1
        else:
            self.dlck_cnt = 0
            self.avoid_dlck_cnt = 0

    def get_next_state(self):
        self.idxs  = self.get_idxs()
        self.count_deadlock()

        # my team closest distance to the ball
        # op team closest distance to the ball
        # deadlock count
        next_state = [self.dist_ball[MY_TEAM][self.idxs[0]], 
                                    self.dist_ball[OP_TEAM].min(), self.dlck_cnt]

        return np.array(next_state)

    def step(self, received_frame):
        self.get_coord(received_frame)
        next_state = self.get_next_state()

        return next_state
##############################################################################
    # function for heuristic coach's action        
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

            if (self.dist_ball[MY_TEAM][robot_id] < 0.5) and (delta[X] > 0):
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

        def set_action(self, act_idx):
            if act_idx == 0:
                # count how many robots is in the goal area
                goal_area_cnt = self.count_goal_area()
                # count how many robots is in the penalty area
                penalty_area_cnt = self.count_penalty_area()

                if goal_area_cnt > 2:
                    avoid_goal_foul(self)
                    self.printConsole('avoid goal foul')
                elif penalty_area_cnt > 3:
                    avoid_penalty_foul(self)
                    self.printConsole('avoid penalty foul')
                else:
                    offense(self)
                    self.printConsole('offense')
            elif act_idx == 1:
                avoid_deadlock(self)
                self.printConsole('avoid deadlock')
            else:
                self.printConsole('action index is over 1')
            
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
            # get next state, reward, and reset info
            next_state = self.step(received_frame)
            next_state = np.reshape(next_state, (1, self.state_dim, 1))

            # # save next state
            self.coach_agent.history = next_state
            self.coach_agent.action = self.coach_agent.\
                        get_action(np.reshape(self.coach_agent.history, (1, -1)))

            # coach's action to positions to go
            set_action(self, self.coach_agent.action)
            set_wheel(self, self.wheels.tolist())
            self.prev_ball = self.cur_ball
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
    
    # create a Wamp session object
    session = Component(ComponentConfig(ai_realm, {}))

    # initialize the msgpack serializer
    serializer = MsgPackSerializer()
    
    # use Wamp-over-rawsocket
    runner = ApplicationRunner(ai_sv, ai_realm, serializers=[serializer])
    
    runner.run(session, auto_reconnect=True)
