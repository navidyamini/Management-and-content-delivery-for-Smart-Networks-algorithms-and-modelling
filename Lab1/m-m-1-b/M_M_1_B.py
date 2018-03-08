import simpy
import random
import numpy
import matplotlib.pyplot as plt
import scipy
import msvcrt as m
import scipy.stats
import pylab
import csv


class confidence_interval(object):
    def __init__(self, data, confidence):
        self.data = data
        self.confidence = confidence

    # @property
    def mean_confidence_interval(self):
        a = 1.0 * numpy.array(self.data)
        n = len(a)
        m = numpy.mean(a)
        std = scipy.stats.sem(a)
        # h = std * scipy.stats.t._ppf((1 + self.confidence) / 2., n - 1)
        new = scipy.stats.t.interval(self.confidence, n - 1, loc=m, scale=std)
        # print h
        # print new
        # return m, m-h, m+h
        return m, new[0], new[1]


class Packets(object):
    """simulating the packets and arrival time queue"""

    def __init__(self, env, arrival_time, min_range, max_range):
        # The inter-arrival time
        self.arrival_time = arrival_time
        # The environment
        self.env = env
        # keepint the current time of the system when batches arrive
        self.batches_current_time_list = []
        self.current_time = 0.0
        # lower band and upper band for creating batches
        self.min_range = min_range
        self.max_range = max_range
        self.batch = 0
        self.bacht_list = []
        self.total_packets = 0

    def packet_arrival(self, server):
        while True:
            # Sample the time to next arrival
            inter_arrival = random.expovariate(lambd=1.0 / self.arrival_time)
            # yield an event to the simulator
            yield self.env.timeout(inter_arrival)
            self.batch = random.randrange(self.min_range, self.max_range)
            self.bacht_list.append(self.batch)
            # keeping the current time for batches
            self.batches_current_time_list.append(self.current_time)
            for i in range(1, self.batch):
                self.total_packets += 1
                self.env.process(server.packet_processing())


class Server(object):
    """ simulating the processing packets in the server"""

    def __init__(self, env, num_server, service_time, buffer_capacity):
        # the service time
        self.service_time = service_time
        # keeping the time for when the service is finished
        self.end_of_service_time_list = []
        # keeping the time when the packet arrive to the queue
        self.starting_of_service_time_list = []
        # keeping the queue size
        self.queue_size_set = []
        self.queue_size_set.append(0)
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
        self.number_of_drroped_packet = 0
        self.buffer_capacity = buffer_capacity

    def packet_processing(self):
        print "\n The number of packets in the queue: %d " % (self.qsize)
        # counting number of packets that arrived to the server
        self.number_of_packets += 1
        print "\n Packet Number %d arrived at %r" % (self.number_of_packets, self.arriving_time)
        self.qsize += 1
        # check if the buffer is full or not
        if (self.qsize > self.buffer_capacity):
            # if the buffer is fuul
            print " buffer is full, packet drroped at: ", env.now
            self.number_of_drroped_packet += 1
            self.qsize -= 1
            self.queue_size_set.append(self.qsize)
        else:  # if the buffer is not full
            # keeping time when the service starts
            self.arriving_time = env.now
            self.starting_of_service_time_list.append(self.arriving_time)
            self.queue_size_set.append(self.qsize)
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
                self.queue_size_set.append(self.qsize)
                print "\n Packet Number %d served at %r " % (self.number_of_served, self.finishing_time)


if __name__ == '__main__':

    MIN_RANGE = 1
    MAX_RANGE = 15
    RANDOM_SEED = 65
    INTER_ARRIVAL = 20.0
    SERVICE_TIME = 3.0
    NUM_SERVER = 1
    SIM_TIME = 1000
    BUFFER_CAPACITY = 45
    CONFIDENCE = 0.95

    counter = 2
    random.seed(RANDOM_SEED)

    roh = []
    warmup_response_time = []
    warmup_buffer = []
    warmup_loss = []
    average_response_time_set = []
    upper_interval_response_time_set = []
    lower_interval_response_time_set = []

    average_buffer_occupancy_set = []
    lower_interval_buffer_occupancy_set = []
    upper_interval_buffer_occupancy_set = []

    number_of_packets_dropped_set = []
    average_packet_loss_set = []
    lower_interval_packet_loss_set = []
    upper_interval_packet_loss_set = []

    total_number_of_packets_set = []
    total_number_of_packets_served = []

    # env = simpy.Environment()
    # packets = Packets(env, INTER_ARRIVAL, MIN_RANGE, MAX_RANGE)
    # server = Server(env, NUM_SERVER, SERVICE_TIME, BUFFER_CAPACITY)
    # env.process(packets.packet_arrival(server))

    # env.run(until=SIM_TIME)
    # for i in range(len(server.end_of_service_time_list)):
    # warmup.append(server.end_of_service_time_list.pop(0) - server.starting_of_service_time_list.pop(0))

    for i in range(145):
        INTER_ARRIVAL -= 0.1
        env = simpy.Environment()
        packets = Packets(env, INTER_ARRIVAL, MIN_RANGE, MAX_RANGE)
        server = Server(env, NUM_SERVER, SERVICE_TIME, BUFFER_CAPACITY)
        env.process(packets.packet_arrival(server))
        # env.run(until=SIM_TIME)# * counter)

        #Removing warp_up
        env.run(until=SIM_TIME)
        for i in range(len(server.end_of_service_time_list)):
            warmup_response_time.append(server.end_of_service_time_list.pop(0) - server.starting_of_service_time_list.pop(0))
            warmup_buffer.append(server.queue_size_set.pop(0) - server.queue_size_set.pop(0))
            #warmup_response_time.append(server.end_of_service_time_list.pop(0) - server.starting_of_service_time_list.pop(0))

        for i in range(10):
            env.run(until=SIM_TIME * counter)
            counter += 1

            # calculating roh
            roh.append(SERVICE_TIME / INTER_ARRIVAL)

            # calculating response time
            response_time = []
            for served, arrived in zip(server.end_of_service_time_list, server.starting_of_service_time_list):
                response_time.append(abs(served - arrived))

                # keeping the number of packets that drroped during simulation
            number_of_packets_dropped_set.append(server.number_of_drroped_packet)
            total_number_of_packets_set.append(packets.total_packets)
            total_number_of_packets_served.append(server.number_of_served)

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

            # calculating average Packet loss
            mean_interval_packet_loss = confidence_interval(number_of_packets_dropped_set, CONFIDENCE)
            average_packet_loss, lower_interval_packet_loss, upper_interval_packet_loss = mean_interval_packet_loss.mean_confidence_interval()
            average_packet_loss_set.append(average_packet_loss)
            lower_interval_packet_loss_set.append(lower_interval_packet_loss)
            upper_interval_packet_loss_set.append(upper_interval_packet_loss)

    with open('average_response_time_M_M_1_B.csv', 'wb') as f:
        rows = zip(lower_interval_response_time_set, average_response_time_set, upper_interval_response_time_set)
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)

    with open('average_buffer_occupancy_M_M_1_B.csv', 'wb') as f:
        rows = zip(lower_interval_buffer_occupancy_set, average_buffer_occupancy_set,
                   upper_interval_buffer_occupancy_set)
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)

    with open('average_packet_loss_M_M_1_B.csv', 'wb') as f:
        rows = zip(lower_interval_packet_loss_set, average_packet_loss_set,
                   upper_interval_packet_loss_set)
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)

    print ("average_response_time: ", average_response_time_set)
    print ("average_buffer_occupancy: ", average_buffer_occupancy_set)
    print ("total_number_of_packets: ", total_number_of_packets_set)
    print ("number_of_packets_dropped: ", number_of_packets_dropped_set)
    print "\n number of packets created during simulation: %r " % (server.number_of_packets)
    print "\n number of packets served by server during simulation: %r" % (server.number_of_served)
    print "\n number of packets stayed in queue after the simulation finished: %r" % (server.qsize)
    print "\n number of packets drroped during simulation: %r " % (server.number_of_drroped_packet)

    # plots
    fig1 = plt.figure(1)
    # response = fig1.add_subplot(111)
    plt.ylabel('response_time')
    plt.plot(roh, average_response_time_set)
    plt.plot(roh, upper_interval_response_time_set, linestyle='--')
    plt.plot(roh, lower_interval_response_time_set, linestyle='--')
    plt.legend(['mean', 'upper_interval', 'lower_interval'], loc='lower right')
    plt.grid(True)
    plt.figure(figsize=(8, 8))
    fig1.show()

    fig2 = plt.figure(2)
    plt.ylabel('buffer_occupancy')
    plt.plot(roh, average_buffer_occupancy_set)
    plt.plot(roh, upper_interval_buffer_occupancy_set, linestyle='--')
    plt.plot(roh, lower_interval_buffer_occupancy_set, linestyle='--')
    plt.legend(['mean', 'upper_interval', 'lower_interval'], loc='lower right')
    plt.grid(True)
    plt.figure(figsize=(8, 8))
    fig2.show()

    fig3 = plt.figure(3)
    plt.ylabel('Packet loss')
    plt.plot(roh, average_packet_loss_set)
    plt.plot(roh, upper_interval_packet_loss_set, linestyle='--')
    plt.plot(roh, lower_interval_packet_loss_set, linestyle='--')
    plt.legend(['mean', 'upper_interval', 'lower_interval'], loc=' lower right')
    plt.grid(True)
    plt.figure(figsize=(8, 8))
    fig3.show()

    pylab.show()
    m.getch()
