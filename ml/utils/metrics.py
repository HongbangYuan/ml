
import time
import logging
from email import message

class Logger(object):
    def __init__(self):
        super(Logger, self).__init__()
        self.item_time = {}
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)

    def log_info(self, message):
        self.log_stdout(message)
        self.log_file(message)

    def log_stdout(self, message):
        print(">>>> {}".format(message))

    def log_file(self, message):
        pass

    def record_item_time(self, item=None):
        if item is not None:
            self.item_time[item] = time.time()
        else:
            self.item_time["no_name"] = time.time()

    def show_item_time(self, item=None):
        if item is not None:
            return time.time() - self.item_time[item]
        else:
            return time.time() - self.item_time["no_name"]

    # @staticmethod
    def record_function(self, func):
        def wrapper(*args, **kwargs):
            self.record_item_time(func.__name__)
            self.log_info('<{}> running'.format(func.__name__))
            ret = func(*args, **kwargs)
            self.log_info(
                "<{}> finished after {:.3f} seconds".format(
                    func.__name__, self.show_item_time(func.__name__)
                )
            )
            # logging.debug(
            #     'finish %s(), cost %s seconds'.format(
            #         func.__name__, self.show_item_time(func.__name__)
            #     )
            # )

            return ret
        return wrapper

def record_function(func):
    def wrapper(*args, **kwargs):
        Logger.record_item_time(item="{}".format(func.__name__))
        Logger.log_info('running %s()' % func.__name__)
        ret = func(*args, **kwargs)
        logging.debug(
            'finish %s(), cost %s seconds'.format(
                func.__name__, Logger.show_item_time(item=func.__name__)
            )
        )

        return ret
    return wrapper