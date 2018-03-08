import simpy
import random
import numpy
import matplotlib.pyplot as plt
import msvcrt as m
import pylab
import scipy.stats
import scipy
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
    def __init__(self, env, arrival_time, min_range, max_range):

        # The inter-arrival time
        self.arrival_time = arrival_time
        self.inter_arrival_list = []
        # The environment
        self.env = env
        # keeping the current time of the system when batches arrive
        self.batches_current_time_list = []
        self.current_time = 0.0
        # lower band and upper band for creating batches
        self.min_range = min_range
        self.max_range = max_range
        self.batch = 0
        self.bacht_list = []

    def packet_arrival(self, server):

        while True:
            # Sample the time to next arrival
            inter_arrival = random.expovariate(lambd=1.0 / self.arrival_time)
            # keeping inter arrival time
            self.inter_arrival_list.append(inter_arrival)
            # yield an event to the simulator
            yield self.env.timeout(inter_arrival)
            self.batch = random.randrange(self.min_range, self.max_range)
            self.bacht_list.append(self.batch)
            self.current_time = env.now
            # keeping the current time for batches
            self.batches_current_time_list.append(self.current_time)
            for i in range(1, self.batch):
                self.env.process(server.packet_processing())


class Front_end_server(object):
    """ simulating the processing packets in the front end server"""

    def __init__(self, env, num_packets, service_time, buffer_capacity, probability, server_2):
        # the service time
        self.service_time = service_time
        self.service_time_list = []
        # keeping the time for when the service is finished
        self.end_of_service_time_list = []
        # keeping the time when the packet arrive to the queue
        self.starting_of_service_time_list = []
        # serve the packet
        self.packets = simpy.Resource(env, num_packets)
        # the environment
        self.env = env
        self.queue_size_set = []
        self.queue_size_set.append(0)
        # number of packets in the ayatem
        self.qsize = 0
        self.number_of_served = 0
        self.arriving_time = 0.0
        self.finishing_time = 0.0
        self.number_of_packets = 0
        self.number_of_drroped_packet = 0
        self.buffer_capacity = buffer_capacity
        self.probability = probability
        self.server_2 = server_2
        self.counter = 0
        self.packets_id_set = []
        self.packet_id_set_finished_here = []

    def packet_processing(self):
        print "\n The number of packets in the front end server's queue : %d " % (self.qsize)
        # counting number of packets that arrived to the server
        self.number_of_packets += 1
        print "\n Packet Number %d arrived at queue # one %r" % (self.number_of_packets, self.arriving_time)
        self.qsize += 1
        # check if the buffer is full or not
        if (self.qsize > self.buffer_capacity):
            # if the buffer is fuul
            print " buffer is full, packet drroped from queue  # one at: ", env.now
            self.number_of_drroped_packet += 1
            self.qsize -= 1
            self.queue_size_set.append(self.qsize)
        else:
            # if the buffer is not full
            # keeping time when the service starts
            self.arriving_time = env.now
            self.starting_of_service_time_list.append(self.arriving_time)
            self.queue_size_set.append(self.qsize)
            with self.packets.request() as request:
                yield request
                service_time = random.expovariate(lambd=1.0 / self.service_time)
                # keeping the service time
                self.service_time_list.append(self.service_time)
                # yield an event to the simulator
                yield self.env.timeout(service_time)
                # keeping the time when the service is finshed
                self.finishing_time = env.now
                self.end_of_service_time_list.append(self.finishing_time)
                # number of served packet in front end server. we can use this number os a id of served packet for calculating response time of whole system
                self.number_of_served += 1
                self.qsize -= 1
                self.queue_size_set.append(self.qsize)
                print "\n %d packets served at the front_end_server %r " % (self.number_of_served, self.finishing_time)
                # self.env.process(server.packet_processing())
                random_number = random.random()
                if (random_number > self.probability):
                    self.packet_id_set_finished_here.append(self.number_of_served)

                else:
                    self.packets_id_set.append(self.number_of_served)
                    self.env.process(self.server_2.packet_processing(self.packets_id_set))
                    # print ("current_time server",current_time)
                    # print ("next arival server: ",service_time)


class Back_end_server(object):
    """ simulating the processing packets in the Back end server"""

    def __init__(self, env, num_packets, service_time_2, buffer_capacity):
        # the service time
        self.service_time = service_time_2
        self.service_time_list = []
        # keeping the time for when the service is finished
        self.end_of_service_time_list = []
        # keeping the time when the packet arrive to the queue
        self.starting_of_service_time_list = []
        # serve the packet
        self.packets = simpy.Resource(env, num_packets)
        # the environment
        self.env = env
        self.queue_size_set = []
        self.queue_size_set.append(0)
        # number of packets in the ayatem
        self.qsize = 0
        self.number_of_served = 0
        self.arriving_time = 0.0
        self.finishing_time = 0.0
        self.number_of_packets = 0
        self.number_of_drroped_packet = 0
        self.buffer_capacity = buffer_capacity
        self.packets_served_number = []

    def packet_processing(self, packets_id_set):
        # def packet_processing(self):
        self.packets_served_number = packets_id_set
        print "\n The number of packets in the backe_end_server's queue: %d " % (self.qsize)
        # counting number of packets that arrived to the server
        self.number_of_packets += 1
        print "\n Packet Number %d arrived at queue # two %r" % (self.number_of_packets, self.arriving_time)
        self.qsize += 1
        # check if the buffer is full or not
        if (self.qsize > self.buffer_capacity):
            # if the buffer is fuul
            print " buffer is full, packet drroped at the queue  # two: ", env.now
            self.number_of_drroped_packet += 1
            self.qsize -= 1
            self.queue_size_set.append(self.qsize)
            self.packets_served_number.pop()
        else:
            # if the buffer is not full
            # keeping time when the service starts
            self.arriving_time = env.now
            self.starting_of_service_time_list.append(self.arriving_time)
            self.queue_size_set.append(self.qsize)
            with self.packets.request() as request:
                yield request
                service_time = random.expovariate(lambd=1.0 / self.service_time)
                # keeping the service time
                self.service_time_list.append(self.service_time)
                # yield an event to the simulator
                yield self.env.timeout(service_time)
                # keeping the time when the service is finshed
                self.finishing_time = env.now
                self.end_of_service_time_list.append(self.finishing_time)
                self.number_of_served += 1
                self.qsize -= 1
                self.queue_size_set.append(self.qsize)
                print "\n  %d packets served at back_end_server %r " % (self.number_of_served, self.finishing_time)


if __name__ == '__main__':

    MIN_RANGE = 1
    MAX_RANGE = 15
    RANDOM_SEED = 64
    INTER_ARRIVAL = 15.5
    BUFFER_CAPACITY_1 = 50
    SERVICE_TIME = 1.0
    BUFFER_CAPACITY_2 = 20
    SERVICE_TIME_2 = 5.0
    PROBABILITY = 0.30
    NUM_SERVER = 1
    SIM_TIME = 100000
    CONFIDENCE = 0.95

    random.seed(RANDOM_SEED)

    average_response_time_1_set = []
    upper_interval_response_time_1_set = []
    lower_interval_response_time_1_set = []

    average_response_time_2_set = []
    upper_interval_response_time_2_set = []
    lower_interval_response_time_2_set = []

    average_total_response_time_set = []
    upper_interval_response_time_set = []
    lower_interval_response_time_set = []

    average_buffer_occupancy_quueq1 = []
    upper_interval_buffer_occupancy_quueq1 = []
    lower_interval_buffer_occupancy_quueq1 = []

    average_buffer_occupancy_quueq2 = []
    upper_interval_buffer_occupancy_quueq2 = []
    lower_interval_buffer_occupancy_quueq2 = []

    roh_server1 = []
    lamda_vector = []
    for i in range(145):

        INTER_ARRIVAL -= 0.1
        env = simpy.Environment()
        server_2 = Back_end_server(env, NUM_SERVER, SERVICE_TIME_2, BUFFER_CAPACITY_2)
        server = Front_end_server(env, NUM_SERVER, SERVICE_TIME, BUFFER_CAPACITY_1, PROBABILITY, server_2)
        packets = Packets(env, INTER_ARRIVAL, MIN_RANGE, MAX_RANGE)
        env.process(packets.packet_arrival(server))
        env.process(server.packet_processing())
        env.run(until=SIM_TIME)


        roh_server1.append(SERVICE_TIME / INTER_ARRIVAL)


        lamda_vector.append(1.0 / INTER_ARRIVAL)

        number_of_packets_1 = server.number_of_packets
        number_of_served_1 = server.number_of_served
        numbers_stayed_in_queue_1 = server.qsize
        number_of_drroped_packets_1 = server.number_of_drroped_packet

        # calculating response time
        response_time_1 = []
        for served, arrived in zip(server.end_of_service_time_list, server.starting_of_service_time_list):
            response_time_1.append(abs(served - arrived))

        number_of_packets_2 = server_2.number_of_packets
        number_of_served_2 = server_2.number_of_served
        numbers_stayed_in_queue_2 = server_2.qsize
        number_of_drroped_packets_2 = server_2.number_of_drroped_packet

        # calculating response time
        response_time_2 = []
        for served, arrived in zip(server_2.end_of_service_time_list, server_2.starting_of_service_time_list):
            response_time_2.append(abs(served - arrived))

        for i in range(0, numbers_stayed_in_queue_2):
            server_2.packets_served_number.pop()

        total_response_time = []
        # adding the response time of those packets that their service has been finished in the front end server
        for j in range(len(server.packet_id_set_finished_here)):
            total_response_time.append(server.end_of_service_time_list[server.packet_id_set_finished_here[j] - 1] -
                                   server.starting_of_service_time_list[server.packet_id_set_finished_here[j] - 1])

        # adding the response time of those packets that their service has been finished in the back end server
        for i in range(len(server_2.packets_served_number)):
            total_response_time.append(server_2.end_of_service_time_list[i] - server.starting_of_service_time_list[
            server_2.packets_served_number[i] - 1])

        # calculating average response time
        mean_interval_response_time = confidence_interval(total_response_time, CONFIDENCE)
        average_response_time, lower_interval_response_time, upper_interval_response_time = mean_interval_response_time.mean_confidence_interval()
        average_total_response_time_set.append(average_response_time)
        upper_interval_response_time_set.append(upper_interval_response_time)
        lower_interval_response_time_set.append(lower_interval_response_time)

        # calculating average buffer1 occupancy
        mean_interval_buffer_occupancy = confidence_interval(server.queue_size_set, CONFIDENCE)
        average_buffer_occupancy, lower_interval_buffer_occupancy, upper_interval_buffer_occupancy = mean_interval_buffer_occupancy.mean_confidence_interval()
        average_buffer_occupancy_quueq1.append(average_buffer_occupancy)
        lower_interval_buffer_occupancy_quueq1.append(lower_interval_buffer_occupancy)
        upper_interval_buffer_occupancy_quueq1.append(upper_interval_buffer_occupancy)

        # calculating average buffer2 occupancy
        mean_interval_buffer_occupancy2 = confidence_interval(server_2.queue_size_set, CONFIDENCE)
        average_buffer_occupancy, lower_interval_buffer_occupancy, upper_interval_buffer_occupancy = mean_interval_buffer_occupancy2.mean_confidence_interval()
        average_buffer_occupancy_quueq2.append(average_buffer_occupancy)
        lower_interval_buffer_occupancy_quueq2.append(lower_interval_buffer_occupancy)
        upper_interval_buffer_occupancy_quueq2.append(upper_interval_buffer_occupancy)

        # calculating average response time queue1
        mean_interval_response_time1 = confidence_interval(response_time_1, CONFIDENCE)
        average_response_time, lower_interval_response_time, upper_interval_response_time = mean_interval_response_time1.mean_confidence_interval()
        average_response_time_1_set.append(average_response_time)
        upper_interval_response_time_1_set.append(upper_interval_response_time)
        lower_interval_response_time_1_set.append(lower_interval_response_time)

        # calculating average response time queue2
        mean_interval_response_time2 = confidence_interval(response_time_2, CONFIDENCE)
        average_response_time, lower_interval_response_time, upper_interval_response_time = mean_interval_response_time2.mean_confidence_interval()
        average_response_time_2_set.append(average_response_time)
        upper_interval_response_time_2_set.append(upper_interval_response_time)
        lower_interval_response_time_2_set.append(lower_interval_response_time)

    with open('average_response_time_ex2.csv', 'wb') as f:
        rows = zip(lower_interval_response_time_set, average_total_response_time_set, upper_interval_response_time_set)
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)

    with open('average_buffer_occupancy_queue1_ex2.csv', 'wb') as f:
        rows = zip(lower_interval_buffer_occupancy_quueq1, average_buffer_occupancy_quueq1,
                   upper_interval_buffer_occupancy_quueq1)
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)

    with open('average_packet_loss_queue2_ex2.csv', 'wb') as f:
        rows = zip(lower_interval_buffer_occupancy_quueq2, average_buffer_occupancy_quueq2,
                   upper_interval_buffer_occupancy_quueq2)
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)

    with open('average_response_time_queue1_ex2.csv', 'wb') as f:
        rows = zip(lower_interval_response_time_1_set, average_response_time_1_set,
               upper_interval_response_time_1_set)
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)

    with open('average_response_time_queue2_ex2.csv', 'wb') as f:
        rows = zip(lower_interval_response_time_2_set, average_response_time_2_set,
                   upper_interval_response_time_2_set)
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)


    # plots
    fig1 = plt.figure(1)
    # response = fig1.add_subplot(111)
    plt.ylabel('response_time_server1')
    plt.plot(roh_server1, average_response_time_1_set)
    plt.plot(roh_server1, upper_interval_response_time_1_set, linestyle='--')
    plt.plot(roh_server1, lower_interval_response_time_1_set, linestyle='--')
    plt.legend(['mean', 'upper_interval', 'lower_interval'], loc='lower right')
    plt.grid(True)
    plt.figure(figsize=(8, 8))
    fig1.show()

    fig2 = plt.figure(2)
    plt.ylabel('response_time_server2')
    plt.plot(lamda_vector, average_response_time_2_set)
    plt.plot(lamda_vector, upper_interval_response_time_2_set, linestyle='--')
    plt.plot(lamda_vector, lower_interval_response_time_2_set, linestyle='--')
    plt.legend(['mean', 'upper_interval', 'lower_interval'], loc='lower right')
    plt.grid(True)
    plt.figure(figsize=(8, 8))
    fig2.show()

    fig3 = plt.figure(3)
    plt.ylabel("total response time for system")
    plt.plot(lamda_vector, average_total_response_time_set)
    plt.plot(lamda_vector, upper_interval_response_time_set, linestyle='--')
    plt.plot(lamda_vector, lower_interval_response_time_set, linestyle='--')
    plt.legend(['mean', 'upper_interval', 'lower_interval'], loc='lower right')
    plt.grid(True)
    plt.figure(figsize=(8, 8))
    fig3.show()

    fig4 = plt.figure(4)
    plt.ylabel('buffer_occupancy_queue1')
    plt.plot(roh_server1, average_buffer_occupancy_quueq1)
    plt.plot(roh_server1, upper_interval_buffer_occupancy_quueq1, linestyle='--')
    plt.plot(roh_server1, lower_interval_buffer_occupancy_quueq1, linestyle='--')
    plt.legend(['mean', 'upper_interval', 'lower_interval'], loc='lower right')
    plt.grid(True)
    plt.figure(figsize=(8, 8))
    fig4.show()

    fig5 = plt.figure(5)
    plt.ylabel('buffer_occupancy_queue2')
    plt.plot(lamda_vector, average_buffer_occupancy_quueq2)
    plt.plot(lamda_vector, upper_interval_buffer_occupancy_quueq2, linestyle='--')
    plt.plot(lamda_vector, lower_interval_buffer_occupancy_quueq2, linestyle='--')
    plt.legend(['mean', 'upper_interval', 'lower_interval'], loc='lower right')
    plt.grid(True)
    plt.figure(figsize=(8, 8))
    fig5.show()

    pylab.show()
    m.getch()