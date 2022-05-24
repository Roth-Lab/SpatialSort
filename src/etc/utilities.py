import datetime
import re
import time


def date_time():
    """
        Returns a dated name, for folder
    """
    dt = datetime.datetime.now().replace(microsecond=0).isoformat()
    pattern = re.compile(r':00\b', flags=re.IGNORECASE)
    folder_name = pattern.sub("", dt).replace("T", "_").replace(":", "").replace("-", "") + "/"
    return folder_name


class Record:
    def __init__(self, t_iter):
        self.t = t_iter
        self.curr_time = time.time()
        self.record = []
        self.interval = 1
        print("Starting Estimation, will print in {} iteration intervals.".format(self.interval))

    def estimate_completion(self, curr, interval=None):
        # Set printing interval if modified
        if interval is not None:
            self.interval = interval

        # Print the time left for this run
        if (curr + 1) % self.interval == 0:
            new_time = time.time()
            elapsed_time = new_time - self.curr_time
            self.curr_time = new_time
            self.record.append(elapsed_time)
            # print("This loop of {} intervals took {} seconds.".format(interval, elapsed_time))
            print("Currently {}% finished, t = {}.".format(round(((curr + 1) / self.t)*100, 2), curr+1))
            seconds = (self.t - 1 - curr) * sum(self.record) / len(self.record) / interval
            print("Estimated Completion: {} minutes.".format(round(seconds/60, 2)))
