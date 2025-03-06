import pybullet as p
import pybullet_data as pd
import time
import math
import numpy as np
import channel
import random
import sys

DEFAULT_PORT=9543
JOINT_FORCE=400
TABLE_HEIGHT=1.0
TABLE_THICKNESS=0.08
TABLE_LENGTH=2.4
TABLE_WIDTH=1.4
DT=1.0/50
INITIAL_CAMERA_ANGLE=90.0
INITIAL_CAMERA_DISTANCE=3.0
NO_DEADLINE=100.0*365.0*86400.0
JOINTS=11
TEXT_HEIGHT=2.5
TEXT_POS=0.5*TABLE_LENGTH+0.5
STOPPED_SPEED_THRESHOLD=0.01
STOPPED_TIME_THRESHOLD=1.0
DIST_THRESHOLD=3*TABLE_LENGTH
STATE_DIMENSION=37
BALL_SERVICE_HEIGHT=TABLE_HEIGHT+1.0 
FONT_SIZE=2.0

class Playfield:
    def __init__(self, game, gui=True):
        self.has_gui=gui
        self.init_pybullet()
        self.load_objects()
        self.cpoints=[]
        self.camera_angle=INITIAL_CAMERA_ANGLE
        self.camera_distance=INITIAL_CAMERA_DISTANCE
        self.finished=False
        self.sim_time=0.0
        self.update_state_cb=None
        self.next_state_cb=None
        self.update_state_deadline=0.0
        self.ball_held_position=None
        self.stopped_time=0.0
        self.game=game
        game.set_playfield(self)
        game.init()

    def load_objects(self):
        self.floor = p.loadURDF('plane.urdf')
        self.table = p.loadURDF('table.urdf', [0, 0, 
                                               TABLE_HEIGHT-TABLE_THICKNESS])
        self.ball  = p.loadURDF('ball.urdf', [0, 0.1, 2],
                                flags=p.URDF_USE_INERTIA_FROM_FILE)
        self.robot = [None, None]
        self.robot[0] = p.loadURDF('robot.urdf',
                                 [0, -0.85-0.5*TABLE_LENGTH, 0.8])
        rot=p.getQuaternionFromEuler([0,0,math.pi])
        self.robot[1] = p.loadURDF('robot2.urdf',

                                 [0, +0.85+0.5*TABLE_LENGTH, 0.8], rot)
        self.objects=[self.floor, self.table, self.ball] + self.robot
        for obj in self.objects:
            for j in range(-1, p.getNumJoints(obj)):
                r=0.95
                if obj==self.table and j==0:
                    r=0.1
                elif (obj==self.robot[0] or obj==self.robot[1]): 
                    if j<=1:
                        r=0.7
                    elif j>=JOINTS-1:
                        r=1.1
                elif obj==self.floor:
                    r=0.7
                p.changeDynamics(obj, j, restitution=r, lateralFriction=3.0)
                if j>=0:
                    p.setJointMotorControl2(obj,j,p.POSITION_CONTROL,
                                        -0.2, force=JOINT_FORCE)
        self.name=["Player 1", "Player 2"]
        self.text=[None, None, None]
        if self.has_gui:
            self.text[0]=p.addUserDebugText(self.name[0],
                    [0, -TEXT_POS, TEXT_HEIGHT], textSize=FONT_SIZE)
            self.text[1]=p.addUserDebugText(self.name[1],
                    [0, +TEXT_POS, TEXT_HEIGHT], textSize=FONT_SIZE)
            self.text[2]=p.addUserDebugText("",
                    [0, 0.0, TEXT_HEIGHT], textSize=FONT_SIZE)

    def update(self):
        self.update_gui()
        self.update_ball()
        if self.update_state_cb:
            if self.sim_time>=self.update_state_deadline:
                self.update_state_cb=None
                if self.next_state_cb:
                    next_cb=self.next_state_cb
                    self.next_state_cb=None
                    next_cb()
            else:
                self.update_state_cb()
        self.game.update()

    def run(self):
        ref_time=time.time()
        try:
          while not self.finished:
            p.stepSimulation()
            cpoints = p.getContactPoints(self.ball)
            if cpoints is not None:
                self.cpoints += cpoints
            p.stepSimulation()
            cpoints = p.getContactPoints(self.ball)
            if cpoints is not None:
                self.cpoints += cpoints
            self.update()
            self.sim_time += DT
            now=time.time()
            dt=ref_time+DT-now
            if dt<=0.0:
                ref_time=now
            else:
                ref_time += DT
                time.sleep(dt)
        finally:
          self.game.on_quit()

    def update_gui(self):
        if not self.has_gui:
            return
        keys=p.getKeyboardEvents()
        if self.pressed(keys, p.B3G_LEFT_ARROW):
            self.camera_angle -= DT*25.0
        if self.pressed(keys, p.B3G_RIGHT_ARROW):
            self.camera_angle += DT*25.0
        if self.pressed(keys, p.B3G_UP_ARROW):
            self.camera_distance=max(1.5, self.camera_distance
                                     -DT*0.5)
        if self.pressed(keys, p.B3G_DOWN_ARROW):
            self.camera_distance=min(5.0, self.camera_distance
                                     +DT*0.5)
        if self.pressed(keys, ord(' ')):
            self.camera_angle=INITIAL_CAMERA_ANGLE
            self.camera_distance=INITIAL_CAMERA_DISTANCE
        p.resetDebugVisualizerCamera(
                cameraDistance=self.camera_distance,
                cameraYaw=self.camera_angle,
                cameraPitch=-30.0,
                cameraTargetPosition=[0.0, 0.0, 1.0])

    def pressed(self, keys, k):
        return keys.get(k, 0) & p.KEY_IS_DOWN

    def update_ball(self):
        if self.ball_held_position:
            p.resetBasePositionAndOrientation(self.ball,
                    self.ball_held_position, [1.0,0.0,0.0,0.0])
        po=p.getBasePositionAndOrientation(self.ball)
        pos=po[0]
        self.ball_position=pos
        dist=math.hypot(pos[0], pos[1])
        self.ball_away=(dist>DIST_THRESHOLD)
        bv=p.getBaseVelocity(self.ball)
        v=bv[0]
        self.ball_velocity=v
        self.ball_speed=math.hypot(v[0], v[1], v[2])
        if self.ball_speed > STOPPED_SPEED_THRESHOLD:
            self.stopped_time=0.0
        else:
            self.stopped_time+=DT
        self.ball_stopped=(self.stopped_time>STOPPED_TIME_THRESHOLD)
        cpoints=self.cpoints
        self.cpoints=[]
        self.contact_floor=False
        self.contact_table=False
        self.contact_robot=[False, False]
        for cp in cpoints:
            if cp[1]!=self.ball:
                continue
            elif cp[2]==self.table and cp[4]==-1:
                self.contact_table=True
            elif cp[2]==self.floor:
                self.contact_floor=True
            elif cp[2]==self.robot[0] and cp[4]>1:
                self.contact_robot[0]=True
            elif cp[2]==self.robot[1] and cp[4]>1:
                self.contact_robot[1]=True
            elif cp[2] in self.robot:
                # cp[4]<=1 
                self.contact_floor=True

    def set_update_state_callback(self, update_cb, next_cb=None, 
                            duration=NO_DEADLINE):
        self.update_state_cb=update_cb
        self.next_state_cb=next_cb
        self.update_state_deadline=self.sim_time+duration

    def init_pybullet(self):
        if not self.has_gui:
            p.connect(p.DIRECT)
        else:
            p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)
        p.setGravity(0,0,-9.81)
        p.setTimeStep(0.5*DT)
        p.setAdditionalSearchPath(pd.getDataPath())

    def schedule_start_positions(self, after_cb=None):
        def next_cb1():
            self.set_update_state_callback(self.start_pos2,
                                           after_cb, 0.9)
        self.set_update_state_callback(self.start_pos1,
                                       next_cb1, 0.7)

    def start_pos1(self):
        jp=[0.0]*JOINTS
        jp[0]=-0.3
        jp[2]=math.pi
        self.set_robot_joints(0, jp)
        self.set_robot_joints(1, jp)

    def start_pos2(self):
        jp=get_neutral_joint_position()
        self.set_robot_joints(0, jp)
        self.set_robot_joints(1, jp)

    def quit(self):
        self.finished=True

    def get_robot_joints(self, index):
        jp=[0.0]*JOINTS
        rob=self.robot[index]
        for j in range(JOINTS):
            jp[j] = p.getJointState(rob, j)[0]
        return jp

    def set_robot_joints(self, index, values):
        rob=self.robot[index]
        for j,val in enumerate(values):
            p.setJointMotorControl2(rob, j, 
                                    p.POSITION_CONTROL,
                                    val,
                                    force=JOINT_FORCE)

    def get_paddle_position_and_normal(self, index):
        rob=self.robot[index]
        pos=[0.0]*3
        nor=[0.0]*3
        ls=p.getLinkState(rob, JOINTS) 
        pos[0:3]=ls[0][0:3]
        quat=ls[1]
        mat=p.getMatrixFromQuaternion(quat)
        nor[0]=mat[2]
        nor[1]=mat[5]
        nor[2]=mat[8]
        return pos,nor


    def set_text(self, index, added=None):
        if not self.has_gui:
            return
        elif index==2:
            text=added if added else ""
        elif added is not None:
            text=self.name[index]+": "+str(added)
        else:
            text=self.name[index]
        pos=0.0 if index==2 else TEXT_POS*(2*index-1)
        z=TEXT_HEIGHT
        if index==2:
            z+=0.5
        p.addUserDebugText(text, [0.0, pos, z],
                           textSize=FONT_SIZE, 
                           replaceItemUniqueId=self.text[index])

    def set_central_text(self, text):
        self.set_text(2, text)

    def set_name(self, index, name):
        self.name[index]=name
        self.set_text(index)

    def hold_ball(self, pos):
        self.ball_held_position=pos

    def throw_ball(self, velocity):
        self.ball_held_position=None
        self.ball_stopped=False
        self.stopped_time=0.0
        p.resetBaseVelocity(self.ball, velocity)

    def get_player_origin(self, index):
        y=index*2.0-1
        return [0.0, y, TABLE_HEIGHT]

    def get_player_direction_y(self, index):
        return 1-index*2



def get_neutral_joint_position():
        jp=[0.0]*JOINTS
        jp[0]=-0.3
        jp[2]=math.pi
        a=math.pi/3.8
        jp[5]=a
        jp[7]=a
        jp[9]=math.pi/3.5 
        jp[10]=math.pi/2
        return jp

class GameDispatcher(channel.Dispatcher):
    def __init__(self, port):
        self.lobby=[]
        super().__init__(port)

    def on_new_channel(self, key, channel):
        name=None
        if type(key)==bytes:
            try:
                name=key.decode('utf8')
            except:
                pass
        if name is None:
            print('=== Unvalid name from connection:', key, '===')
            return
        with self.lock:
            self.lobby.append((name, channel))
    
    def get_next(self):
        with self.lock:
            if not self.lobby:
                return None
            item=self.lobby[0]
            del self.lobby[0]
            return item


class Game:
    def __init__(self):
        self.dispatcher=None
        self.num_players=0
        self.player=[None, None]
        self.player_name=[None, None]
        self.player_active=[False, False]
        self.score=[0, 0]
        self.field_touch=False
        self.robot_touch=False
        self.concerned_player=-1
        self.waiting=True
        self.waiting_service=False
        self.serving_player=0
        self.game_started=False
        self.game_time=0.0
        self.sched_queue=[]
        self.sched_time=0.0
        self.reason=None
        self.score_limit=None
        self.time_limit=None

    def set_playfield(self, playfield):
        self.playfield=playfield

    def set_score_limit(self, limit):
        self.score_limit=limit
    
    def set_time_limit(self, limit):
        self.time_limit=limit

    def enable_dispatcher(self, port=DEFAULT_PORT):
        self.dispatcher=GameDispatcher(port)

    def swap_serving_player(self):
        self.serving_player=1-self.serving_player

    def init(self):
        pf=self.playfield
        pf.hold_ball([0,0,2.0])
        pf.schedule_start_positions()
        pf.set_central_text('waiting for players')
        print('=== Waiting for players ===')

    def update(self):
        pf=self.playfield
        self.update_schedule()
        if self.dispatcher:
            self.update_dispatcher()
        self.update_play()
        self.prepare_state()
        for index in [0, 1]:
            if self.player[index]:
                s=self.compute_state(index)
                jp=self.player[index].update(s)
                if self.player_active[index]:
                    pf.set_robot_joints(index, jp)
        if self.game_started:
            t=int(self.game_time)
            ss=t%60
            mm=t//60
            msg='%02d:%02d' % (mm, ss)
            pf.set_text(2, msg)
            self.game_time += DT

    def update_dispatcher(self):
        if self.num_players==2:
            return
        item=self.dispatcher.get_next()
        if not item:
            return
        name, channel=item
        player=RemotePlayerInterface(channel, name)
        self.add_player(player, name)

    def prepare_state(self):
        pf=self.playfield
        self.joints=[pf.get_robot_joints(0), 
                     pf.get_robot_joints(1)]
        self.paddle=[pf.get_paddle_position_and_normal(0),
                      pf.get_paddle_position_and_normal(1)]
        x,y,z=self.paddle[1][0]
        dx,dy,dz=self.paddle[1][1]
        tx=x+0.6*dx
        ty=y+0.6*dy
        tz=z+0.6*dz



    def compute_state(self, index):
        pf=self.playfield
        y_dir=2*index-1
        state=[0.0]*STATE_DIMENSION
        state[0:10]=self.joints[index]
        pad=self.paddle[index]
        state[11:14]=self.convert_coordinates(index, pad[0])
        state[14:17]=self.convert_vector(index, pad[1], +1)
        state[17:20]=self.convert_coordinates(index, pf.ball_position)
        state[20:23]=self.convert_vector(index, pf.ball_velocity)
        opad=self.paddle[1-index]
        state[23:26]=self.convert_coordinates(index, opad[0])
        oserve=self.waiting_service and self.serving_player!=index
        waiting=self.waiting or (self.waiting_service and not oserve)
        state[26]=int(waiting)
        state[27]=int(oserve)
        state[28]=int(not self.waiting and not self.waiting_service)
        state[29]=int(pf.ball_position[1]*y_dir > 0.0)
        if self.concerned_player==index:
            state[30]=int(self.field_touch)
            state[31]=int(self.robot_touch)
            state[33]=0
        else:
            state[30]=0
            state[31]=0
            state[33]=int(self.field_touch)
        state[32]=1-state[29]
        state[34]=self.score[index]
        state[35]=self.score[1-index]
        state[36]=self.game_time
        return state

    def convert_coordinates(self, index, vec):
        orig=self.playfield.get_player_origin(index)
        if index==0:
            return vec[0]-orig[0], vec[1]-orig[1], vec[2]-orig[2]
        else:
            return orig[0]-vec[0], orig[1]-vec[1], vec[2]-orig[2]

    def convert_vector(self, index, vec, desired_y_sign=None):
        x, y, z=vec
        if index==1:
            x=-x
            y=-y
        if desired_y_sign:
            if y*desired_y_sign<0.0:
                x=-x
                y=-y
                z=-z
        return x,y,z

    def on_quit(self):
        for p in self.player:
            if p is not None:
                p.on_quit()
        if self.dispatcher:
            self.dispatcher.shutdown()

    def add_player(self, player, name):
        print('=== Adding player:', name,' ===')
        np=self.num_players
        if np>=2:
            self.playfield.quit()
            print('*** TOO MANY PLAYERS ***')
            return
        pf=self.playfield
        self.num_players=np+1
        self.player[np]=player
        self.player_name[np]=name
        pf.set_name(np, name)
        if np==1:
            pf.set_central_text("")
            self.schedule(self.on_ready)

    def update_schedule(self):
        t=self.sched_time
        while self.sched_queue and self.sched_queue[0][0]<=t:
            callback=self.sched_queue[0][1]
            del self.sched_queue[0]
            callback()
        self.sched_time += DT

    def schedule(self, callback, delay=0.0):
        t=self.sched_time + delay
        i=0
        n=len(self.sched_queue)
        while i<n and self.sched_queue[i][0]<=t:
            i+=1
        self.sched_queue.insert(i, (t, callback))

    def get_service_position(self, index):
        pf=self.playfield
        pos=pf.get_player_origin(index)
        pos[0]+=random.uniform(-0.1, 0.1)
        pos[2]=BALL_SERVICE_HEIGHT
        return pos

    def get_service_velocity(self, index):
        pf=self.playfield
        dy=pf.get_player_direction_y(index)
        x0, y0, z0=pf.ball_position
        dl=abs(y0)+0.25*TABLE_LENGTH+random.uniform(-0.1, 0.3)
        dh=z0-TABLE_HEIGHT
        g=9.81
        v=dl*math.sqrt(g/(2*(dh+dl)))
        vz=v
        vx=random.uniform(-0.2*v, 0.2*v)
        vy=v*dy
        vv=math.hypot(vx, vy)/v
        vx=vx/vv
        vy=vy/vv
        return [vx, vy, vz]

    def update_play(self):
        if self.waiting or self.waiting_service:
            return
        if self.time_limit and self.game_time>=self.time_limit:
            self.waiting=True
            self.reason='time limit was reached'
            self.schedule(self.on_terminate)
            return
        pf=self.playfield
        by=pf.ball_position[1]
        if pf.contact_floor or pf.ball_away or pf.ball_stopped:
            self.waiting=True
            if self.concerned_player>=0:
                self.schedule(self.on_score_point)
            else:
                self.schedule(self.on_restart, 1.0)
            return
        for index in [0,1]:
            dy=pf.get_player_direction_y(index)
            if pf.contact_robot[index]:
                if by*dy>0.001:
                    self.concerned_player=index
                    self.waiting=True
                    self.schedule(self.on_score_point)
                    return
                if self.concerned_player==index and \
                        self.robot_touch:
                    self.waiting=True
                    self.schedule(self.on_score_point)
                    return
                if self.concerned_player==index:
                    self.robot_touch=True
                else:
                    self.concerned_player=index
                    self.robot_touch=True
                    self.field_touch=False
            if pf.contact_table and by*dy<-0.001:
                if self.concerned_player==index: 
                    self.waiting=True
                    self.schedule(self.on_score_point)
                    return
                self.concerned_player=index
                self.robot_touch=False
                self.field_touch=True
                    

        

    def on_ready(self):
        print('=== Ready to start ===')
        pf=self.playfield
        pf.set_text(0, self.score[0])
        pf.set_text(1, self.score[1])
        self.game_started=True
        self.schedule(self.on_prepare_service)

    def on_prepare_service(self):
        index=self.serving_player
        print('=== Preparing for service:', self.player_name[index],
              '===')
        pf=self.playfield
        self.player_active=[False, False]
        self.waiting=True
        self.waiting_service=False
        pf.hold_ball(self.get_service_position(index))
        pf.schedule_start_positions(self.on_ready_to_serve)

    def on_ready_to_serve(self):
        index=self.serving_player
        self.waiting=False
        self.waiting_service=True
        self.player_active[1-index]=True
        self.field_touch=False
        self.robot_touch=False
        self.concerned_player=-1
        dt=random.uniform(0.5, 1.0)
        self.schedule(self.on_serve_ball, dt)

    def on_serve_ball(self):
        self.waiting=False
        self.waiting_service=False
        index=self.serving_player
        self.player_active[index]=True
        pf=self.playfield
        v=self.get_service_velocity(index)
        pf.throw_ball(v) 
        

    def on_score_point(self):
        self.waiting=True
        player_index=1-self.concerned_player
        print('=== Point scored for player: ',
              self.player_name[player_index], '===')
        self.score[player_index]+=1
        pf=self.playfield
        pf.set_text(player_index, self.score[player_index])
        if self.score_limit and \
                self.score[player_index]>=self.score_limit:
            self.reason='score limit was reached'
            self.schedule(self.on_terminate, 0.5)
        else:
            self.schedule(self.on_restart, 0.5)
            
    def on_restart(self):
        self.waiting=True
        self.swap_serving_player()
        self.schedule(self.on_prepare_service, 0.1)

    def on_terminate(self):
        reason=''
        if self.reason:
            reason='because '+self.reason
        print('=== Game terminated', reason,'===')
        print('=== Total game time:', self.game_time, '===')
        print('=== Final score: ===')
        print('  ', self.score[0], 'points for', self.player_name[0])
        print('  ', self.score[1], 'points for', self.player_name[1])
        self.playfield.quit()




class NoBallGame(Game):
    def update_play(self):
        pass

    def set_score_limit(self, limit):
        print('--- Warning: score limit ignored ---')

    def on_serve_ball(self):
        self.waiting=False
        self.waiting_service=False
        index=self.serving_player
        self.player_active[index]=True

    def get_service_position(self, index):
        return [0, 0, 3.0]

class SameServeGame(Game):
    def on_restart(self):
        self.waiting=True
        self.schedule(self.on_prepare_service, 0.1)

class NormalGame(Game):
    pass


class PlayerInterface:
    def update(self, state):
        pass

    def on_quit(self):
        pass

class DummyPlayerInterface(PlayerInterface):
    def update(self, state):
        jp=get_neutral_joint_position()
        jp[1]=state[17]
        return jp

class RemotePlayerInterface(PlayerInterface):
    def __init__(self, channel, name):
        self.channel=channel
        self.name=name
        self.last_joints=get_neutral_joint_position()

    def update(self, state):
        self.send(state)
        return self.receive()

    def on_quit(self):
        self.channel.close()

    def send(self, state):
        msg=channel.encode_float_list(state)
        self.channel.send(msg)

    def receive(self):
        last_msg=self.channel.receive()
        if last_msg is None:
            return self.last_joints
        msg=self.channel.receive()
        while msg is not None:
            last_msg=msg
            msg=self.channel.receive()
        jp=channel.decode_float_list(last_msg)
        if jp is None or len(jp)!=JOINTS:
            print('** Received bad message from', self.name, '**')
        else:
            self.last_joints=jp
        return self.last_joints



class AutoPlayerInterface(PlayerInterface):
    def __init__(self):
        jp=get_neutral_joint_position()
        self.stance=[
                self.make_stance(0.3, 0, 0.3, 0.3, -0.6-0.2),
                self.make_stance(0.3, 2.7, -1.3, -1.8, +0.4-0.1),
                self.make_stance(0.3, 0.4, -1, -1.3, +2)]
        self.stance_height=[0.17, 0.17, 0.6]
        self.stance_dy=[-0.32, +0.08, -0.35]
        self.chosen_stance=0
        self.freeze_stance=0.0

    def make_stance(self, a, j3, j5, j7, j9):
        jp=get_neutral_joint_position()
        jp[3] += a*j3
        jp[5] += a*j5
        jp[7] += a*j7
        jp[9] += a*j9 
        return jp

    def update(self, state):
        self.freeze_stance-=DT
        px, py, pz=state[11:14]
        bx, by, bz=state[17:20]
        vx, vy, vz=state[20:23]
        if vy>0 or state[31]:
            jp=self.stance[self.chosen_stance].copy()
            jp[1]=bx
            return jp
        self.choose_stance(state)
        jp=self.stance[self.chosen_stance].copy()
        #dy=self.stance_dy[self.chosen_stance]
        self.choose_position(state, jp)
        return jp

    def choose_stance(self, state):
        px, py, pz=state[11:14]
        bx, by, bz=state[17:20]
        vx, vy, vz=state[20:23]
        dist=math.hypot(px-bx, py-by, pz-bz)
        if not state[28] or dist<0.1 or self.freeze_stance>0.0:
            return
        g=9.81
        d=0.05
        minz=bz
        while by>-0.5 and bz>0.0:
            by+=vy*d
            bz+=vz*d
            vz-=g*d
            minz=min(minz,bz)
        curr=self.chosen_stance
        if minz>0.35:
            self.chosen_stance=2
        elif by>0 and by<0.30 and not state[30]:
            self.chosen_stance=1
        elif by<TABLE_LENGTH*0.5 and by>0.90 and not state[30]:
            self.chosen_stance=1
        elif by>0 and state[30]:
            self.chosen_stance=1
        elif by<-0.2:
            self.chosen_stance=0
        elif curr==2:
            self.chosen_stance=0
        if self.chosen_stance!=curr:
            self.freeze_stance=0.75

    def choose_position(self, state, jp):
        px, py, pz=state[11:14]
        bx, by, bz=state[17:20]
        vx, vy, vz=state[20:23]
        dist=math.hypot(px-bx, py-by, pz-bz)
        vel=math.hypot(vx, vy, vz)
        if state[27] or vel<0.05:
            jp[1]=bx
            return
        extra_y=0.0
        if dist<vel*1.5*DT:
            extra_y=0.3
        d=0.05
        g=9.81
        while vz>0 or bz+d*vz>=pz:
            bx+=d*vx
            by+=d*vy
            bz+=d*vz
            vz-=d*g
        jp[1]=bx
        dy=py-state[0]
        jp[0]=by-dy+extra_y
        jp[10]-=(bx*0.3+vx*0.01)

class Options:
    def __init__(self, **kwargs):
        for k in kwargs:
            self.__dict__[k]=kwargs[k]

def parse_time(s):
    f=s.split(':')
    if len(f)<1 or len(f)>2:
        print('*** Unvalid time:', s)
        sys.exit(1)
    if len(f)==1:
        return int(f[0])
    else:
        return int(f[1])+60*int(f[0])

def parse_options():
    opt=Options(port=DEFAULT_PORT,
              time=None,
              score=None,
              game=NormalGame,
              swap=False,
              gui=True,
              font=1.0,
              players=[])
    n=len(sys.argv)
    i=1
    while i<n:
        a=sys.argv[i]
        while a[:1]=='-':
            a=a[1:]
        if a=='port':
            i+=1
            opt.port=int(sys.argv[i])
        elif a=='time':
            i+=1
            opt.time=parse_time(sys.argv[i])
        elif a=='score':
            i+=1
            opt.score=int(sys.argv[i])
        elif a=='noball':
            opt.game=NoBallGame
        elif a=='sameserve':
            opt.game=SameServeGame
        elif a=='swap':
            opt.swap=True
        elif a=='dummy':
            opt.players.append(DummyPlayerInterface)
        elif a=='auto':
            opt.players.append(AutoPlayerInterface)
        elif a=='nogui':
            opt.gui=False
        elif a=='font':
            i+=1
            opt.font=float(sys.argv[i])
        else:
            print('*** Unvalid command line option:', sys.argv[i])
            sys.exit(1)
        i+=1
    return opt


def main():
    global FONT_SIZE
    opt=parse_options()
    ga=opt.game()
    pf=Playfield(ga, opt.gui)
    if opt.time is not None:
        ga.set_time_limit(opt.time)
    if opt.score is not None:
        ga.set_score_limit(opt.score)
    if opt.swap:
        ga.swap_serving_player()
    FONT_SIZE*=opt.font
    i=0
    for p in opt.players:
        i+=1
        name='Player %d'%(i)
        ga.add_player(p(), name)
    ga.enable_dispatcher(opt.port)
    pf.run()

if __name__=='__main__':
    main()
