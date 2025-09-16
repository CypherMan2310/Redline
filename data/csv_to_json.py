import csv
import json


def csv_to_json ( csvFilePath, jsonFilePath ) :
    jsonArray = []

    # read csv file
    with open( csvFilePath, encoding='utf-8', errors='ignore' ) as csvf :
        # load csv file data using csv library's dictionary reader
        csvReader = csv.DictReader( csvf )

        # convert each csv row into python dict
        for row in csvReader :
            # add this python dict to json array
            jsonArray.append( row )

    # convert python jsonArray to JSON String and write to file
    with open( jsonFilePath, 'w', encoding='utf-8' ) as jsonf :
        jsonString = json.dumps( jsonArray, indent=4 )
        jsonf.write( jsonString )

csvFilePath = r"245453521061489663873_table_1.csv"
jsonFilePath = r"245453521061489663873_table_1.json"
csv_to_json(csvFilePath, jsonFilePath)
#Completion Message
print("Conversion completed successfully")