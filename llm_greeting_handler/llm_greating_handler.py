import sys
sys.path.append('gen-py')

from thrift import Thrift
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer
from LlmGreetingHandler import LlmGreeting


class LlmGreatingHandler():
    def __init__(self):
        self.log = ()
    
    def generate_greeting(self, message: str):
        # call large language model
        pass


if __name__ == "__main__":
    handler = LlmGreatingHandler()
    processor = LlmGreeting.Processor(handler)
    transport = TSocket.TServerSocket(host='127.0.0.1', port=9090)
    tfactory = TTransport.TBufferedTransportFactory()
    pfactory = TBinaryProtocol.TBinaryProtocolFactory()

    server = TServer.TSimpleServer(processor, transport, tfactory, pfactory)
    print("Starting the server...")
    server.serve()
    print("done.")