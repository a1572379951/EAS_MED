import numpy as np
import fixed_env as env
import load_trace
import itertools


S_INFO = 5  # bit_rate, buffer_size, rebuffering_time, bandwidth_measurement, chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 8
MPC_FUTURE_CHUNK_COUNT = 5
VIDEO_BIT_RATE = [400, 1000, 2000, 3500, 5300, 7500, 9500, 12000]
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 150.0
TOTAL_VIDEO_CHUNKS = 150
TAU = 20.0
GAMMA = 0.6
M_IN_K = 1000.0
REBUF_PENALTY = 6  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000000
SUMMARY_DIR = './PSQA_result'
LOG_FILE = './PSQA_result/PSQA'
# log in format of time_stamp bit_rate buffer_size rebuffer_time chunk_size download_time reward
# NN_MODEL = './models/nn_model_ep_5900.ckpt'

CHUNK_COMBO_OPTIONS = []

# past errors in bandwidth
past_errors = []
past_bandwidth_ests = []

def get_chunk_size(quality, index):
    if ( index < 0 or index > 150 ):
        return 0

    chunk_size = VIDEO_BIT_RATE[quality]*250
    return chunk_size


def main():

    np.random.seed(RANDOM_SEED)

    assert len(VIDEO_BIT_RATE) == A_DIM

    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace("./test_file/")

    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw)

    log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
    log_file = open(log_path, 'w')

    time_stamp = 0

    last_bit_rate = DEFAULT_QUALITY
    bit_rate = DEFAULT_QUALITY

    action_vec = np.zeros(A_DIM)
    action_vec[bit_rate] = 1

    s_batch = [np.zeros((S_INFO, S_LEN))]
    a_batch = [action_vec]
    r_batch = []
    entropy_record = []

    video_count = 0



    while True:  # serve video forever
        # the action is from the last decision
        # this is to make the framework similar to the real
        delay,sleep_time, buffer_size, rebuf, video_chunk_size, next_video_chunk_sizes, end_of_video, video_chunk_remain = net_env.get_video_chunk(bit_rate)

        time_stamp += delay  # in ms
        time_stamp += sleep_time  # in ms

        # reward is video quality - rebuffer penalty
        reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                 - REBUF_PENALTY * rebuf \
                 - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                           VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K


        r_batch.append(reward)

        last_bit_rate = bit_rate

        log_file.write(str(time_stamp / M_IN_K) + '\t' +
                       str(VIDEO_BIT_RATE[bit_rate]) + '\t' +
                       str(buffer_size) + '\t' +
                       str(rebuf) + '\t' +
                       str(video_chunk_size) + '\t' +
                       str(delay) + '\t' +
                       str(reward) + '\n')
        log_file.flush()


        if len(s_batch) == 0:
            state = [np.zeros((S_INFO, S_LEN))]
        else:
            state = np.array(s_batch[-1], copy=True)

        # dequeue history record
        state = np.roll(state, -1, axis=1)

        # this should be S_INFO number of terms
        state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))  # last quality
        state[1, -1] = buffer_size / BUFFER_NORM_FACTOR
        state[2, -1] = rebuf
        state[3, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
        state[4, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)


        past_bandwidths = state[3, -5:]

        while past_bandwidths[0] == 0.0:
            if len(past_bandwidths) <= 1:
                break
            past_bandwidths = past_bandwidths[1:]

        bandwidth_sum = 0
        for past_val in past_bandwidths:
            bandwidth_sum += (1/float(past_val+0.001))
        harmonic_bandwidth = 1.0/(bandwidth_sum/len(past_bandwidths))

        harmonic_bandwidth = harmonic_bandwidth * 8000
        past_bandwidth_ests.append(harmonic_bandwidth)

        bit_rate = GAMMA * harmonic_bandwidth * (buffer_size + TAU) / TAU

        min_difference = 10000000

        curr_bitrate = bit_rate
        for index, target_bitrate in enumerate(VIDEO_BIT_RATE):
            difference = abs(curr_bitrate - target_bitrate)
            if difference < min_difference:
                min_difference = difference
                bit_rate = index

        s_batch.append(state)

        if end_of_video:
            log_file.write('\n')
            log_file.close()

            last_bit_rate = DEFAULT_QUALITY
            bit_rate = DEFAULT_QUALITY  # use the default action here

            del s_batch[:]
            del a_batch[:]
            del r_batch[:]

            action_vec = np.zeros(A_DIM)
            action_vec[bit_rate] = 1

            s_batch.append(np.zeros((S_INFO, S_LEN)))
            a_batch.append(action_vec)
            entropy_record = []

            print("video count", video_count)
            video_count += 1

            if video_count >= len(all_file_names):
                break

            log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
            log_file = open(log_path, 'w')


if __name__ == '__main__':
    main()

