import csv
# Read .YUV file and process channels
def read_YUV_file(file_path):
    # Read .YUV file and extract Y, U, V channels
    # Process channels to extract edge detection values, vertical and horizontal speeds
    return y_channel, u_channel, v_channel

# Process YUV channels and combine with hand location coordinates
def process_YUV_channels(y_channel, u_channel, v_channel, hand_location):
    # Process YUV channels and adjust values to center point 0
    edge_detection_values = y_channel - 128
    vertical_speeds = u_channel - 128
    horizontal_speeds = v_channel - 128
    
    # Combine with hand location coordinates
    combined_data = []
    for i in range(len(hand_location)):
        timestamp = hand_location[i][0]  # Timestamp is the first element in hand_location
        x_left, y_left, x_right, y_right = hand_location[i][1:]  # Hand location coordinates follow timestamp
        combined_data.append([timestamp, x_left, y_left, x_right, y_right, vertical_speeds[i], horizontal_speeds[i]])
    
    return combined_data
def load_object_detection_results(csv_file):
    """
    Load object detection results from a CSV file.
    Parameters:
        csv_file (str): Path to the CSV file containing object detection results.
    Returns:
        list: A list of tuples, where each tuple represents the object detection results for a single frame.
              Each tuple contains the timestamp and the coordinates of the left and right hand.
    """
    object_detection_results = []

    with open(csv_file, mode='r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header row

        for row in csv_reader:
            timestamp = row[0]  # Timestamp is the first column
            left_hand_x, left_hand_y = float(row[1]), float(row[2])  # Left hand coordinates are in columns 2 and 3
            right_hand_x, right_hand_y = float(row[3]), float(row[4])  # Right hand coordinates are in columns 4 and 5
            object_detection_results.append((timestamp, left_hand_x, left_hand_y, right_hand_x, right_hand_y))

    return object_detection_results
# Main function
def main(opt):

    csv_file = "C:/Users/home/Desktop/Camera_recognition/TimeForecast/hand_coordinates_20240305-225002.csv"

    # Load object detection results
    object_detection_results = load_object_detection_results()  # Loads the results into a suitable data structure
    
    # Read .YUV file
    y_channel, u_channel, v_channel = read_YUV_file(yuv_file_path)  # Provide path to .YUV file
    
    # Process YUV channels and combine with hand location coordinates
    combined_data = process_YUV_channels(y_channel, u_channel, v_channel, object_detection_results)
    
