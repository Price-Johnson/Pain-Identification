import fileIO
import os
import participant

def main():
    print(os.listdir())
    fileIO.readGarminParticipantData("raw_data/raw_data_garmin/participant_1_BL.csv")

if __name__ == "__main__":
    main()
