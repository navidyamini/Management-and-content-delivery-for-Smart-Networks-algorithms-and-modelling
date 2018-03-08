import simpy
import random
import numpy
import matplotlib.pyplot as plt
import scipy.stats
import scipy
import msvcrt as m
import pylab
import csv

class confidence_interval(object):
    def __init__(self, data, confidence):
        self.data = data
        self.confidence = confidence

    #@property
    def mean_confidence_interval(self):
        a = 1.0 * numpy.array(self.data)
        n = len(a)
        m = numpy.mean(a)
        std = scipy.stats.sem(a)
        #h = std * scipy.stats.t._ppf((1 + self.confidence) / 2., n - 1)
        new = scipy.stats.t.interval(self.confidence, n - 1, loc= m, scale=std)
        #print h
        #print new
        #return m, m-h, m+h
        return m, new[0], new[1]

class Packets(object):
    """simulating the packets and arrival time queue"""
    def __init__(self, env, arrival_time):
        # The inter-arrival time
        self.arrival_time = arrival_time
        # The environment
        self.env = env

    def packet_arrival(self, server):
        while True:
            # Sample the time to next arrival
            inter_arrival = random.expovariate(lambd=1.0 / self.arrival_time)
            # yield an event to the simulator
            yield self.env.timeout(inter_arrival)
            self.env.process(server.packet_processing())

class Server(object):
    """ simulating the processing packets in the server"""
    def __init__(self, env, num_server, service_time):
        # the service time
        self.service_time = service_time
        # keeping the time for when the service is finished
        self.end_of_service_time_list = []
        # keeping the time when the packet arrive to the queue
        self.starting_of_service_time_list = []
        # keeping the queue size
        self.queue_size_set = []
        # serve the packet
        self.packets = simpy.Resource(env, num_server)
        # the environment
        self.env = env
        # number of packets in the ayatem
        self.qsize = 0
        self.number_of_served = 0
        self.arriving_time = 0.0
        self.finishing_time = 0.0
        self.number_of_packets = 0

    def packet_processing(self):
        print "\n The number of packets in the queue: %d " % (self.qsize)
        self.arriving_time = env.now
        # keeping time when the service starts
        self.starting_of_service_time_list.append(self.arriving_time)
        # counting number of packets that arrived to the server
        self.number_of_packets += 1
        print "\n Packet Number %d arrived at %r" % (self.number_of_packets, self.arriving_time)
        self.queue_size_set.append(self.qsize)
        self.qsize += 1
        with self.packets.request() as request:
            yield request
            service_time = random.expovariate(lambd=1.0 / self.service_time)
            # yield an event to the simulator
            yield self.env.timeout(service_time)
            # keeping the time when the service is finshed
            self.finishing_time = env.now
            self.end_of_service_time_list.append(self.finishing_time)
            self.number_of_served += 1
            self.qsize -= 1
            print "\n Packet Number %d served at %r " % (self.number_of_served, self.finishing_time)

if __name__ == '__main__':

    RANDOM_SEED = 40
    INTER_ARRIVAL = 15.0
    SERVICE_TIME = 0.1
    NUM_SERVER = 1
    SIM_TIME = 1000000
    CONFIDENCE = 0.95

    average_time_in_system = []
    average_buffer_size_in_system=[]
    average_response_time_set = []
    upper_interval_response_time_set = []
    lower_interval_response_time_set = []

    average_buffer_occupancy_set = []
    lower_interval_buffer_occupancy_set = []
    upper_interval_buffer_occupancy_set = []

    roh=[]

    random.seed(RANDOM_SEED)

    for i in range(145):
        SERVICE_TIME += 0.1

        #for calculating theory part
        lamda = 1.0/INTER_ARRIVAL
        mu = 1.0/SERVICE_TIME
        ET= 1.0/(mu-lamda)
        average_time_in_system.append(ET)
        average_buffer_size_in_system.append(average_time_in_system[-1]/INTER_ARRIVAL)

        env = simpy.Environment()
        packets = Packets(env, INTER_ARRIVAL)
        server = Server(env, NUM_SERVER, SERVICE_TIME)
        env.process(packets.packet_arrival(server))

        env.run(until=SIM_TIME)

        # calculating roh
        roh.append(SERVICE_TIME / INTER_ARRIVAL)

        # calculating response time
        response_time = []
        for served, arrived in zip(server.end_of_service_time_list, server.starting_of_service_time_list):
            response_time.append(served - arrived)

        # calculating average response time
        mean_interval_response_time = confidence_interval(response_time, CONFIDENCE)
        average_response_time, lower_interval_response_time, upper_interval_response_time = mean_interval_response_time.mean_confidence_interval()
        average_response_time_set.append(average_response_time)
        upper_interval_response_time_set.append(upper_interval_response_time)
        lower_interval_response_time_set.append(lower_interval_response_time)


        # calculating average buffer occupancy
        mean_interval_buffer_occupancy = confidence_interval(server.queue_size_set, CONFIDENCE)
        average_buffer_occupancy, lower_interval_buffer_occupancy, upper_interval_buffer_occupancy = mean_interval_buffer_occupancy.mean_confidence_interval()
        average_buffer_occupancy_set.append(average_buffer_occupancy)
        lower_interval_buffer_occupancy_set.append(lower_interval_buffer_occupancy)
        upper_interval_buffer_occupancy_set.append(upper_interval_buffer_occupancy)


    with open('average_response_time_M_M_1.csv', 'wb') as f:
        rows = zip(lower_interval_response_time_set, average_response_time_set, upper_interval_response_time_set)
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)
    with open('average_buffer_occupancy_M_M_1.csv', 'wb') as f:
        rows = zip(lower_interval_buffer_occupancy_set, average_buffer_occupancy_set,
                   upper_interval_buffer_occupancy_set)
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)
    print "\naverage_response_time: %r " %(average_response_time_set)
    print "\naverage_buffer_occupancy: %r " %(average_buffer_occupancy_set)
    print "\n number of packets created during simulation: %r " % (server.number_of_packets)
    print "\n number of packets served by server during simulation: %r" % (server.number_of_served)
    print "\n number of packets stayed in queue after the simulation finished: %r" % (server.qsize)

    # plots
    fig1 = plt.figure(1)
    plt.ylabel('response_time')
    plt.plot(roh, average_time_in_system)
    plt.plot(roh, average_response_time_set)
    plt.plot(roh, upper_interval_response_time_set, linestyle='--')
    plt.plot(roh, lower_interval_response_time_set, linestyle='--')

    plt.legend(['theoretical response time','mean', 'upper_interval', 'lower_interval'], loc='upper left')
    plt.grid(True)
    plt.figure(figsize=(8, 8))
    fig1.show()
    #plt.waitforbuttonpress()

    fig2 = plt.figure(2)
    plt.ylabel('buffer_occupancy')
    plt.plot(roh, average_buffer_size_in_system)
    plt.plot(roh, average_buffer_occupancy_set)
    plt.plot(roh, upper_interval_buffer_occupancy_set, linestyle='--')
    plt.plot(roh, lower_interval_buffer_occupancy_set, linestyle='--')
    plt.legend(['theoretical queue size','mean', 'upper_interval', 'lower_interval'], loc='upper left')
    plt.grid(True)
    plt.figure(figsize=(8, 8))
    fig2.show()
    pylab.show()
    #plt.waitforbuttonpress()

    m.getch()
