class Participant:

    def __init__(self, id, user_id, unix_in_ms, ibi):
        self.id = id
        self.user_id = user_id
        self.unix_in_ms = unix_in_ms
        self.ibi = ibi

    def printParticipantInformation(self):
        print("Participant " + self.id + ":\n User ID: " + self.user_id +
              "\n Unix In Miliseconds: " + self.unix_in_ms +
              "\n IBI: " + self.ibi + "\n")

    # getter and setter methods

    # id getter method
    def getID(self):
        return self.id

    # user_id getter method
    def getUser_ID(self):
        return self.user_id

    # unix_in_ms getter method
    def getUnix_in_ms(self):
        return self.unix_in_ms

    # ibi getter method
    def getIbi(self):
        return self.ibi

    # id setter method
    def setID(self, id):
        self.id = id

    # user_id setter method
    def setUser_ID(self, user_id):
        self.user_id = user_id

    # unix_in_ms setter method
    def setUnix_in_ms(self, unix_in_ms):
        self.unix_in_ms = unix_in_ms

    # ibi setter method
    def setIbi(self, ibi):
        self.ibi = ibi
