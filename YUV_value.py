def write_yuv_to_text(y, u, v, frame_number):
    # Write Y values
    with open(f'y_values_frame_{frame_number}.txt', 'w') as f:
        for value in y:
            f.write(f"{value}\n")

    # Write U values
    with open(f'u_values_frame_{frame_number}.txt', 'w') as f:
        for value in u:
            f.write(f"{value}\n")

    # Write V values
    with open(f'v_values_frame_{frame_number}.txt', 'w') as f:
        for value in v:
            f.write(f"{value}\n")

def read_frame_yuv420p(file, width, height):
    frame_size = width * height + (width // 2) * (height // 2) * 2
    y_size = width * height
    uv_size = (width // 2) * (height // 2)

    frame_data = file.read(frame_size)
    if not frame_data:
        return None

    y = frame_data[:y_size]
    u = frame_data[y_size:y_size+uv_size]
    v = frame_data[y_size+uv_size:]

    return y, u, v

# Example usage
frame_number = 1  # Starting with the first frame
with open('output.yuv', 'rb') as f:
    width, height = 640, 480  # Adjust to your video's resolution
    while True:
        frame_yuv = read_frame_yuv420p(f, width, height)
        if frame_yuv:
            y, u, v = frame_yuv
            write_yuv_to_text(y, u, v, frame_number)
            print(f"Frame {frame_number} processed.")
            frame_number += 1
        else:
            break  # End of file
