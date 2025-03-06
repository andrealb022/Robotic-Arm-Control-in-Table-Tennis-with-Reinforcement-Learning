import socket
import time
import queue
import threading
import select
import struct
from abc import ABC, abstractmethod

MESSAGE_NORMAL = 128
MAX_MESSAGE_LENGTH=65535-MESSAGE_NORMAL
MESSAGE_PING   = 1
MESSAGE_REFUSED = 2
MESSAGE_HEADER_SIZE=2
MAX_OUTBOUND_QUEUE = 15
MIN_OUTBOUND_BUFFER = 4096
RECV_SIZE=MAX_MESSAGE_LENGTH+MESSAGE_HEADER_SIZE
WAIT_TIME=0.2
PING_TIME=2.0


class ChannelError(IOError):
    pass

class AbstractChannel(ABC):
    @abstractmethod
    def send(self, message):
        pass

    @abstractmethod
    def send_refuse(self):
        pass

    @abstractmethod
    def receive(self, timeout=None):
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def is_closed(self):
        pass

    @abstractmethod
    def is_refused(self):
        pass

    @abstractmethod
    def is_finished(self):
        pass

    @abstractmethod
    def last_activity_time(self):
        pass


class BaseChannel(AbstractChannel):
    def __init__(self):
        self.lock = threading.RLock()
        self.sock = None
        self.inbound_queue = queue.SimpleQueue()
        self.outbound_queue = queue.SimpleQueue()
        self.inbound_buffer = b''
        self.outbound_buffer = b''
        self.closed = False
        self.last_write_time = time.time()
        self.last_read_time = time.time()
        self.found_error=False
        self.refused = False
        self.reader_finished = False
        self.writer_finished = False
        t1=threading.Thread(target=self.reader_thread)
        t1.start()
        t2=threading.Thread(target=self.writer_thread)
        t2.start()

    def set_socket(self, sock):
        with self.lock:
            try:
                if self.sock:
                    self.sock.close()
            finally:
                self.sock=sock
            self.inbound_buffer = b''
            self.outbound_buffer = b''

    def send(self, message):
        with self.lock:
            if self.closed:
                return
            if self.refused:
                raise ChannelError('Send to a connection refused by peer')
        if len(message)>MAX_MESSAGE_LENGTH:
                raise ChannelError('Message too long in send')
        try:
            while self.outbound_queue.qsize()>MAX_OUTBOUND_QUEUE:
                self.outbound_queue.get(block=False)
        except queue.Empty:
            pass
        self.outbound_queue.put(message)

    def send_refuse(self):
        m=struct.pack('!H', MESSAGE_REFUSED)
        with self.lock:
            self.outbound_buffer += m
            

    def receive(self, timeout=None):
        with self.lock:
            if self.refused:
                raise ChannelError('Receive from a connection refused by peer')
            if self.closed:
                return None
        try:
            if not timeout:
                return self.inbound_queue.get(block=False)
            elif timeout>=0.0:
                return self.inbound_queue.get(timeout=timeout)
            else:
                return self.inbound_queue.get()
        except queue.Empty:
            return None

    def close(self):
        with self.lock:
            self.closed=True

    def is_closed(self):
        with self.lock:
            return self.closed

    def is_refused(self):
        with self.lock:
            return self.refused

    def is_finished(self):
        with self.lock:
            return self.reader_finished and self.writer_finished 

    def last_activity_time(self):
        with self.lock:
            return max(self.last_read_time, self.last_write_time)

    def reader_thread(self):
        while not self.is_closed():
            fd=None
            with self.lock:
                if self.sock:
                    fd=self.sock.fileno()
            if fd is None:
                time.sleep(WAIT_TIME)
                continue
            rready, _, xready=select.select([fd],[],[fd],WAIT_TIME)
            with self.lock:
                if not self.sock or fd!=self.sock.fileno():
                    continue
                if xready:
                    self.found_error=True
                if rready and not self.found_error:
                    self.do_read()
            self.check_error()
            # End of while
        with self.lock:
            self.reader_finished=True
            try:
                if self.writer_finished and self.sock:
                    self.sock.close()
            except IOError:
                pass

    def writer_thread(self):
        while not self.is_closed():
            self.prepare_write()
            fd=None
            with self.lock:
                if self.sock:
                    fd=self.sock.fileno()
            if fd is None:
                time.sleep(WAIT_TIME)
                continue
            _, wready, xready=select.select([],[fd],[fd],WAIT_TIME)
            with self.lock:
                if not self.sock or fd!=self.sock.fileno():
                    continue
                if xready:
                    self.found_error=True
                if wready and not self.found_error:
                    self.do_write()
            self.check_error()
            # End of while
        with self.lock:
            self.writer_finished=True
            try:
                if self.reader_finished and self.sock:
                    self.sock.close()
            except IOError:
                pass



    def prepare_write(self):
        with self.lock:
            bl=len(self.outbound_buffer)
        while bl<MIN_OUTBOUND_BUFFER:
            try:
                if bl==0:
                    m=self.outbound_queue.get(timeout=PING_TIME)
                else:
                    m=self.outbound_queue.get(block=False)
            except queue.Empty:
                break
            em=self.encode_message(m)
            with self.lock:
                self.outbound_buffer += em
                bl=len(self.outbound_buffer)
        if bl==0:
            ping=struct.pack('!H', MESSAGE_PING)
            with self.lock:
                self.outbound_buffer+=ping

    def encode_message(self, message):
        n=len(message)
        if n>MAX_MESSAGE_LENGTH:
            print('*** Warning: Attempted sending a message too long!')
            return ''
        h=struct.pack('!H', MESSAGE_NORMAL+n)
        return h+message


    def do_write(self):
        if self.outbound_buffer:
            try:
                n=self.sock.send(self.outbound_buffer)
            except IOError:
                n=-1
            if n<1:
                self.found_error=True
            else:
                self.outbound_buffer = self.outbound_buffer[n:]
                self.last_write_time=time.time()

    def do_read(self):
        data=b''
        try:
            data=self.sock.recv(RECV_SIZE)
        except IOError:
            self.found_error=True
        if data:
            self.inbound_buffer += data
            self.last_read_time=time.time()
            self.parse_messages()
        else:
            self.found_error=True

    def check_error(self):
        with self.lock:
            if self.found_error:
                self.found_error=False
                try:
                    self.sock.close()
                finally:
                    self.sock=None
                self.on_error()


    def on_error(self):
        pass

    def post_message(self, message):
        self.inbound_queue.put(message)

    def parse_messages(self):
        while True:
            blen=len(self.inbound_buffer)
            if blen<MESSAGE_HEADER_SIZE:
                break
            header=self.inbound_buffer[:MESSAGE_HEADER_SIZE]
            msg_code=struct.unpack('!H', header)[0]
            msg_length=max(msg_code-MESSAGE_NORMAL, 0)
            msg_code=msg_code - msg_length
            if msg_code==MESSAGE_NORMAL:
                if blen<MESSAGE_HEADER_SIZE+msg_length:
                    break
                message=self.inbound_buffer[MESSAGE_HEADER_SIZE:
                                            MESSAGE_HEADER_SIZE+msg_length]
                self.inbound_buffer=self.inbound_buffer[
                                            MESSAGE_HEADER_SIZE+msg_length:]
                self.post_message(message)
            elif msg_code==MESSAGE_PING:
                self.inbound_buffer=self.inbound_buffer[MESSAGE_HEADER_SIZE:]
            elif msg_code==MESSAGE_REFUSED:
                self.refused=True
                self.close()
                break
            else:
                self.found_error=True
                break




class TransientChannel(BaseChannel):
    def __init__(self, sock, dispatcher):
        self.dispatcher=dispatcher
        self.received_first=False
        super().__init__()
        self.set_socket(sock)

    def close(self):
        super().close()
        self.dispatcher.close_channel(self)

    def on_error(self):
        self.close()

    def post_message(self, message):
        with self.lock:
            if self.received_first:
                super().post_message(message)
                return
            else:
                self.received_first=True
        self.dispatcher.register_channel(self, message)


class ClientChannel(BaseChannel):
    def __init__(self, host, port, hello_message):
        self.host=host
        self.port=port
        self.hello_message=hello_message
        super().__init__()
        self.reconnect()

    def on_error(self):
        self.reconnect()

    def reconnect(self):
        if self.is_refused():
            return
        with self.lock:
            if self.sock:
                try:
                    self.sock.close()
                finally:
                    self.sock=None
            n=0
            while not self.sock:
                try:
                    self.sock=create_client_socket(self.host, self.port)
                except IOError:
                    n=min(n+1, 50)
                    time.sleep(0.1*n)
            self.inbound_buffer=b''
            self.outbound_buffer=self.encode_message(self.hello_message)


class ServerChannel(AbstractChannel):
    def __init__(self, key, dispatcher):
        self.key=key
        self.dispatcher=dispatcher
        self.delegate=None
        self.lock=threading.RLock()
        self.closed=False
        self.last_time=time.time()

    def get_key(self):
        return self.key

    def set_delegate(self, channel):
        with self.lock:
            if self.delegate:
                self.delegate.close()
            self.delegate=channel
            self.last_time=time.time()
            if self.closed and self.delegate:
                self.delegate.close()
                self.delegate=None
                

    def send(self, message):
        with self.lock:
            if self.delegate:
                self.delegate.send(message)

    def send_refuse(self):
        with self.lock:
            if self.delegate:
                self.delegate.send_refuse()

    def receive(self, timeout=None):
        with self.lock:
            if self.delegate:
                return self.delegate.receive(timeout)
            else:
                return None

    def close(self):
        with self.lock:
            if self.delegate:
                self.delegate.close()
                self.delegate=None
            self.closed=True
        self.dispatcher.close_channel(self)

    def is_closed(self):
        with self.lock:
            return self.closed

    def is_refused(self):
        with self.lock:
            if self.delegate:
                return self.delegate.is_refused()
        return False

    def is_finished(self):
        return self.is_closed()

    def last_activity_time(self):
        with self.lock:
            if self.delegate:
                return self.delegate.last_activity_time()
            else:
                return self.last_time


class Dispatcher:
    def __init__(self, port):
        self.sock=create_server_socket(port)
        self.must_finish=False
        self.finished=False
        self.error=False
        self.lock=threading.RLock()
        self.transient_channels=set()
        self.server_channels=dict()
        t=threading.Thread(target=self.thread_function)
        t.start()



    def connection_key(self, hello_message):
        return hello_message

    def register_channel(self, transient_channel, hello_message):
        key=self.connection_key(hello_message)
        if key is None:
            transient_channel.send_refuse()
            self.close_channel(transient_channel)
        else:
            with self.lock:
                if key in self.server_channels:
                    self.server_channels[key].set_delegate(transient_channel)
                else:
                    sc=ServerChannel(key, self)
                    sc.set_delegate(transient_channel)
                    self.server_channels[key]=sc
                    self.on_new_channel(key, sc)


    def on_new_channel(self, key, channel):
        pass

    def on_error(self):
        pass

    def close_channel(self, channel):
        if isinstance(channel, TransientChannel):
            with self.lock:
                if channel in self.transient_channels:
                    self.transient_channels.remove(channel)
        else:
            key=channel.get_key()
            with self.lock:
                if key in self.server_channels:
                    del self.server_channels[key]
        if not channel.is_closed():
            channel.close()

    def get_keys(self):
        with self.lock:
            return set(self.server_channels.keys())

    def get_channel(self, key):
        with self.lock:
            return self.server_channels.get(key, None) 


    def shutdown(self, wait=False):
        with self.lock:
            self.must_finish=True
        if wait:
            while not self.is_finished():
                time.sleep(WAIT_TIME)

    def is_finished(self):
        with self.lock:
            return self.finished

    def is_in_error(self):
        with self.lock:
            return self.error

    def thread_function(self):
        with self.lock:
            must_finish=self.must_finish
        while not must_finish:
            with self.lock:
                fd=self.sock.fileno()
            rready,_,xready=select.select([fd],[],[fd], WAIT_TIME)
            if xready:
                with self.lock:
                    self.must_finish=True
                    self.error=True
                    self.on_error()
            if rready:
                with self.lock:
                    try:
                        if not self.must_finish:
                            sock,addr=self.sock.accept()
                            tc=TransientChannel(sock, self)
                            self.transient_channels.add(tc)
                    except IOError:
                        self.must_finish=True
                        self.error=True
                        self.on_error()
            with self.lock:
                must_finish=self.must_finish
            # End While
        try:
            self.sock.close()
        except IOError as e:
            pass
        self.sock=None
        with self.lock:
            for ch in list(self.transient_channels):
                ch.close()
            for ch in list(self.server_channels.values()):
                ch.close()
            self.transient_channels.clear()
            self.server_channels.clear()
            self.finished=True


def create_server_socket(port):
    return socket.create_server(('',port))


def create_client_socket(host, port):
    return socket.create_connection((host,port))

def encode_float_list(lst):
    try:
        return b''.join((struct.pack('!f', x) for x in lst))
    except:
        return None

def decode_float_list(msg):
    try:
        return list(x[0] for x in struct.iter_unpack('!f', msg))
    except:
        return None
