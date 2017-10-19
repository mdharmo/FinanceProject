import scanf as s
import csv

InputFileName = "/home/mharmon/FinanceProject/Data/2015-12-11-mdp_book_builder_output.log"
csvname = 'cmebook' + '11' + '.csv'


stuff = csv.writer(open(csvname,'wb'))

#print ", ".join(
#    ["sent time", "received time", "delta or aggressor", "type", "SSN", "ISN", "bid depth", "bid volume", "bid price",
#     "ask depth", "ask volume", "ask price"])
with open(InputFileName) as input_file:
    index_line = 0
    formated_str = ""

    oneRow = []

    for line in input_file:

        if line == "\n":
            if oneRow != []:
                #print oneRow
                stuff.writerow(oneRow)
            index_line = 1
            continue

        if index_line == 0:
            pass
        elif index_line == 1:


            oneRow = []
            num_segments = len(line.split(" "))
            if line.split()[0][2]==':':
                mysplit = line.split()
                header =  mysplit[0]+mysplit[1]+mysplit[2]

            elif num_segments == 9:
                header1, header2, header3, header4, ssn, isn, sent, recv, indx = s.sscanf(line,
                                                                                          "%s %s %s %s SSN:%s ISN:%s Sent:%s Received:%s (%d)")
                header = header1 + header2 + header3 + header4
            elif num_segments == 7:
                header1, header2, ssn, isn, sent, recv, indx = s.sscanf(line, "%s %s SSN:%s ISN:%s Sent:%s Received:%s (%d)")
                header = header1 + header2
            elif num_segments == 15:
                header1, header2, header3, header4, ssn, isn, sent, recv, indx = s.sscanf(line,
                                                                                          "%s %s %s %s SSN:%s ISN:%s Sent:%s Received:%s %s")
                indx = " ".join(line.split()[-7:])
                header = header1 + header2 + header3 + header4
            elif num_segments == 14:

                header1, header2, header3, header4, ssn, isn, sent, recv, indx = s.sscanf(line,
                                                                                          "%s %s %s %s SSN:%s ISN:%s Sent:%s Received:%s %s")
                indx = " ".join(line.split()[-6:])
                header = header1 + header2 + header3 + header4
            elif num_segments == 13:
                header1, header2, ssn, isn, sent, recv, indx = s.sscanf(line, "%s %s SSN:%s ISN:%s Sent:%s Received:%s %s")
                indx = " ".join(line.split()[-7:])
                header = header1 + header2
            elif num_segments == 12:
                header1, header2, ssn, isn, sent, recv, indx = s.sscanf(line, "%s %s SSN:%s ISN:%s Sent:%s Received:%s %s")
                indx = " ".join(line.split()[-6:])
                header = header1 + header2
            #print ", ".join([sent, recv, str(indx), header, ssn, isn, '']),
            # add to oneRow
            #print header1,header2
            oneRow += [header,sent,recv]
        else:
            #print line.split()
            # This actually extracts the real data
            elements = line.split()
            field1 = elements[1][:-1]
            field2 = elements[2]
            field3, field4 = elements[4].split('|')
            field5 = elements[6]
            field6 = elements[7][1:]

            oneRow += [field1,field2,field3,field4,field5,field6]
        index_line += 1

