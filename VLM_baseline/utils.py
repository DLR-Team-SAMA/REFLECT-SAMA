import os
class ConvLogger:
    def __init__(self, log_folder):
        # create log folder if not exist
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)
        
        self.log_text = ''

        # check log_no file exists else create it
        if not os.path.exists(log_folder+'log_no.txt'):
            with open(log_folder+'log_no.txt', 'w') as f:
                f.write('0')

        with open(log_folder+'log_no.txt', 'r') as f:
            self.log_no = int(f.read())
        self.log_no += 1
        with open(log_folder+'log_no.txt', 'w') as f:
            f.write(str(self.log_no))
        self.log_fn = log_folder+'log_'+str(self.log_no)+'.txt'
        
    def save(self):
        self.log = open(self.log_fn, 'w')
    
    def log_append(self, text):
        self.log_text += text+'\n'
