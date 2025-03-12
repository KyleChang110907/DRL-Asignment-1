import gym
import pickle
import random
import numpy as np
import imageio  # used for saving GIFs
from PIL import Image, ImageDraw, ImageFont, ImageOps

def text_to_image(text, font=None, bg_color="white", text_color="black", fixed_size=None):
    """
    Convert a text string (from env.render()) into a PIL Image.
    Uses a monospaced font and pads each line to the same width for a uniform grid.
    If fixed_size is provided as (width, height), the generated image is padded/resized to that size.
    """
    # Use a monospaced font if not provided.
    if font is None:
        try:
            font = ImageFont.truetype("DejaVuSansMono.ttf", 16)
        except Exception as e:
            font = ImageFont.load_default()
    lines = text.split("\n")
    # Pad each line to the same length.
    max_length = max(len(line) for line in lines)
    padded_lines = [line.ljust(max_length) for line in lines]
    
    # Calculate natural width and height for each padded line.
    widths, heights = [], []
    for line in padded_lines:
        print(line)
        bbox = font.getbbox(line)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        widths.append(width)
        heights.append(height)
    
    natural_width = max(widths) if widths else 0
    natural_height = sum(heights) if heights else 0
    
    # Create an image with natural size.
    img = Image.new("RGB", (natural_width, natural_height), color=bg_color)
    draw = ImageDraw.Draw(img)
    y = 0
    for i, line in enumerate(padded_lines):
        draw.text((0, y), line, fill=text_color, font=font)
        y += heights[i]
    
    # If a fixed_size is provided, pad/resize the image to match it.
    if fixed_size is not None:
        img = ImageOps.pad(img, fixed_size, color=bg_color)
    return img

def test_agent(num_episodes=5, q_table_file="./results/q_table.pkl"):
    # Load the trained Q-table from file.
    try:
        with open(q_table_file, "rb") as f:
            q_table = pickle.load(f)
        print("Q-table loaded successfully from", q_table_file)
    except Exception as e:
        print("Error loading Q-table:", e)
        return

    # Define a local get_action that uses the loaded Q-table.
    def local_get_action(state):
        # For Taxi-v3 the state is an integer.
        if state in q_table:
            return int(np.argmax(q_table[state]))
        else:
            return random.choice(range(6))  # fallback if state is missing

    # Create the Taxi-v3 environment with render_mode "ansi" to obtain text output.
    env = gym.make("Taxi-v3", render_mode="ansi")
    rewards = []
    fixed_size = None  # Will be determined by the first captured frame

    # Define a nested function to annotate a PIL image.
    def annotate_frame(img, state, dest_text, font=None, bg_color="white", text_color="black"):
        """
        Append two extra lines to the bottom of img:
          - The first line is the passenger location (color) for the given state.
          - The second line is the destination for the episode.
        """
        if font is None:
            font = ImageFont.load_default()
        # Decode the state to get passenger location.
        taxi_row, taxi_col, pass_loc, _ = env.decode(state)
        passenger_mapping = {0: "Red", 1: "Green", 2: "Yellow", 3: "Blue", 4: "In taxi"}
        pass_text = passenger_mapping.get(pass_loc, "")
        print(f"Passenger location: {pass_text}")
        # Determine heights for the two extra text lines.
        bbox1 = font.getbbox(pass_text)
        line1_height = bbox1[3] - bbox1[1]
        bbox2 = font.getbbox(dest_text)
        line2_height = bbox2[3] - bbox2[1]
        extra_height = line1_height + line2_height
        
        # Create a new image with extra space at the bottom.
        width, height = img.size
        new_img = Image.new("RGB", (width, height + extra_height), color=bg_color)
        new_img.paste(img, (0, 0))
        draw = ImageDraw.Draw(new_img)
        # Draw the passenger location (first extra line) and destination (second extra line).
        draw.text((0, height), pass_text, fill=text_color, font=font)
        draw.text((0, height + line1_height), dest_text, fill=text_color, font=font)
        return new_img

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        # Handle Gym API returning (observation, info)
        if isinstance(state, tuple):
            state = state[0]
        done = False
        total_reward = 0
        frames = []  # list to store frames for this episode
        
        # Determine destination for this episode using the initial state.
        _, _, _, dest_idx = env.decode(state)
        destination_mapping = {0: "Red", 1: "Green", 2: "Yellow", 3: "Blue"}
        dest_text = destination_mapping.get(dest_idx, "")
        print(f"Episode {episode}: Destination = {dest_text}")

        # Capture the initial frame (text output converted to image) and annotate it.
        frame_text = env.render()  # returns ANSI text
        frame_image = text_to_image(frame_text)
        # Set fixed_size based on the first frame, if not already set.
        if fixed_size is None:
            fixed_size = frame_image.size
        frame_image = text_to_image(frame_text, fixed_size=fixed_size)
        # Annotate the frame with passenger location and destination.
        annotated_image = annotate_frame(frame_image, state, dest_text)
        frames.append(np.array(annotated_image))

        while not done:
            action = local_get_action(state)
            next_state, reward, done, _, info = env.step(action)
            if isinstance(next_state, tuple):
                next_state = next_state[0]
            total_reward += reward
            state = next_state
            
            # Capture the frame after the step.
            frame_text = env.render()  # returns ANSI text
            frame_image = text_to_image(frame_text, fixed_size=fixed_size)
            annotated_image = annotate_frame(frame_image, state, dest_text)
            frames.append(np.array(annotated_image))
            print(f'step:{len(frames)}')
        rewards.append(total_reward)
        print(f"Episode {episode}: Total Reward = {total_reward}")
        
        # Save the frames as a GIF for this episode.
        gif_filename = f"episode_{episode}.gif"
        imageio.mimsave(gif_filename, frames, fps=2)  # Adjust fps as needed

    env.close()
    return rewards

if __name__ == "__main__":
    test_agent(num_episodes=5)
