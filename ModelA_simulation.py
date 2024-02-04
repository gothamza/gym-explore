
import gymnasium as gym
import streamlit as st
import numpy as np
from PIL import Image
import time
import os
from DQNAgent import DQNAgent
import matplotlib.pyplot as plt
import os
from PIL import ImageOps
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# Function to run training and display results
def run_training(selected_game, selected_model, epochs,parameters):
    losses = []
    rewards = []

    # Load the selected game environment
    env = gym.make(f"ALE/{selected_game}", render_mode='rgb_array')
    env.metadata["render_fps"] = 500
    # Initialize the DQNAgent
    
    if selected_model == "DQN":
        agent = DQNAgent(env.observation_space.shape, env.action_space.n, lr=int(parameters['learning_rate']), gama=parameters['gamma'],memory_size=parameters['memory_size'],batch_size =parameters['batch_size'],epsilon=1,chkpt_file=parameters['save_name'])
        if keep_simulation :
            agent.load_file(nammod)
    
    # Reset the environment to get the initial state
    state, _ = env.reset()

    # Directory to save frames as images
    frames_dir = "frames"
    os.makedirs(frames_dir, exist_ok=True)
    
    image_placeholder = st.empty()
    # Run the training loop
    for epoch in range(1, epochs + 1):
        done = False
        total_reward = 0.0
        state, _ = env.reset()
        loss_t = 0.0
        reward_t = 0.0
        lifes = env.unwrapped.ale.lives()

        while not done:
            action = agent.act(state)
            state_, reward, done, _, _ = env.step(action)
            current_lifes = env.unwrapped.ale.lives()

            # Punish for dying
            if current_lifes < lifes:
                lifes = current_lifes
                reward_t -= 50

            # Actual reward
            reward_t += reward
            # For time
            reward_t -= 1

            agent.remember(state, action, reward, state_, done)
            loss_t += agent.learn()

            # Render the current state as an image
            # Render the current state as an image
            frame = env.render()

            # Save the frame as an image
            image = Image.fromarray(frame)

            # Add a border to the image
            border_color = 'black'  # Change this to the color you want
            border_width = 10  # Change this to the border width you want
            image = ImageOps.expand(image, border=border_width, fill=border_color)

            # Resize the image
            width = 200  # Change this to the width you want
            height = 200  # Change this to the height you want
            image = image.resize((width, height))

            # Display the image in Streamlit
            image_placeholder.image(image, caption=f"Action: {action}", use_column_width=True)

            # Add a small delay to display each image
            time.sleep(0.001)

        agent.be_reasonable(epoch)
        losses.append(loss_t)
        rewards.append(reward_t)

        st.text(f"******** loss at epoch {epoch} = {loss_t} ********")
        st.text(f"******** reward at epoch {epoch} = {reward_t} ********")
        print(f"******** loss at epoch {epoch} = {loss_t} ********")
        print(f"******** reward at epoch {epoch} = {reward_t} ********")
        if end_simulation :  
            break 
        
    print(f"--------- saving the model at epoch {epoch} ---------")
    agent.save()
            
        
        
    return losses, rewards


# Streamlit app
st.title("Reinforcement Learning Training")

# User interface for selecting game, model, and epochs
selected_game = st.selectbox("Select Gym Game", ["SpaceInvaders-v5", "CartPole-v5", "Pong-v5", "BeamRider-v5", "Breakout-v5", "Enduro-v5", "Qbert-v5", "Seaquest-v5"])
selected_model = st.selectbox("Select Model", [ "PPO","DQN", "ModelB"])
epochs = st.slider("Number of Epochs", min_value=5, max_value=200, value=20, step=1)
parameters = {}

if selected_model == "DQN":
    st.sidebar.title("DQN Parameter Selection")
    lr_model = st.sidebar.selectbox("Learning Rate", [0.05, 0.01])
    gamma_model = st.sidebar.selectbox("Gamma Parameter", [0.99, 0.9])
    memory_model = st.sidebar.selectbox("Memory Parameter", [1000, 1100])
    batch_model = st.sidebar.selectbox("Batch Size Parameter", [32, 132])
    namepth = st.sidebar.text_input("Enter your saving name for this model") 

    # Add parameters to the dictionary
    parameters['learning_rate'] = lr_model
    parameters['gamma'] = gamma_model
    parameters['memory_size'] = memory_model
    parameters['batch_size'] = batch_model
    parameters['save_name'] = str(namepth )+".pth"

    #lr=0.05,gama=0.99,memory_size=1000,batch_size = 32,epsilon=1,chkpt_file="DQN.pth"
# Button to start the simulation
nammod = st.text_input("Enter your saved model name if u want to keep training ")
nammod= str(nammod )+".pth"
keep_simulation=False
if (nammod != ".pth" ) :
    keep_simulation = st.button("Start trained model ")


start_simulation = st.button("Start Training")




# ...
import csv
import time

# Function to convert seconds to hours, minutes, and seconds
def convert_seconds_to_hms(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return int(hours), int(minutes), int(seconds)

# ...if keep_simulation :



if start_simulation or keep_simulation:
    # Record the start time
    end_simulation = st.button("stop and save ")
    print(parameters['save_name'])
    start_time = time.time()

    # Run training and get results
    losses, rewards = run_training(selected_game, selected_model, epochs, parameters)

    # Record the end time
    end_time = time.time()
    
    # Calculate the training duration
    training_duration_seconds = end_time - start_time
    training_duration_hours, training_duration_minutes, training_duration_seconds = convert_seconds_to_hms(training_duration_seconds)

    # Save losses, rewards, and training duration to CSV file
    csv_filename = "training_results.csv"

    # Check if the CSV file exists
    if not os.path.exists(csv_filename):
        # If it doesn't exist, create a new CSV file with headers
        with open(csv_filename, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            header = [ 'Save Name','Game', 'Model', 'Epoch','Training Duration', 'Learning Rate', 'Gamma', 'Memory Size', 'Batch Size']
            for i in range(5):
                header.extend([f'Loss{i + 1}', f'Reward{i + 1}'])  # Loss1, Reward1, ..., Loss5, Reward5
            
            csv_writer.writerow(header)

    # Append current results to CSV file
    with open(csv_filename, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        row = [parameters['save_name'],selected_game, selected_model, epochs,f"{training_duration_hours}H {training_duration_minutes}Min {training_duration_seconds}S", parameters['learning_rate'],
               parameters['gamma'], parameters['memory_size'], parameters['batch_size']]
        for epoch, (loss, reward) in enumerate(zip(losses, rewards), start=1):
            save_interval = epochs // 5
            if epoch % save_interval == 0:
                row.extend([loss, reward])
        csv_writer.writerow(row)

    # ... Rest of the code remains unchanged ...






    # Display the training plots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

    # Plot on the left subplot
    axes[0].plot(range(epochs), losses, label='loss')
    axes[0].set_title('Loss per Epoch')
    axes[0].legend()

    # Plot on the right subplot
    axes[1].plot(range(epochs), rewards, label='reward', color='orange')
    axes[1].set_title('Reward per Epoch')
    axes[1].legend()

    # Adjust layout for better spacing
    plt.tight_layout()

    # Show the plots in Streamlit
    st.pyplot(fig)

    # Optionally, you can display other information or save the plots as images.
    st.success(f"Training and saving completed! You can now select another game and model.")
