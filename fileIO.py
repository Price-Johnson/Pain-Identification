import csv
import sys


def readGarminParticipantData(fileName):
    with open(fileName, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
            print(f'\t{row["id"]} {row["user_id"]} {row["unix_in_ms"]} {row["ibi"]}.')
            line_count += 1
        print(f'Processed {line_count} lines.')
