def fitness_function_pt(multitree, num_episodes=5, episode_duration=300, render=False, ignore_done=False):
  memory = ReplayMemory(10000)
  rewards = []

  for _ in range(num_episodes):
    # get initial state of the environment
    observation = env.reset()
    observation = observation[0]
    action = np.array([0,0])

    for _ in range(episode_duration):
      if render:
        frames.append(env.render())

      input_sample = torch.from_numpy(observation.reshape((1,-1))).float()
      
      # Improvement symmetry
      # Mirror the input, and use that as the input
      input = observation.reshape((1,-1))       
      outputs = multitree.get_output_pt(torch.from_numpy(transform_input(input))) # Get the outputs for the mirrored input, does this work? --W
      print(outputs.shape)

      # !!! Transform the output from [1, 4] dimension tensor to [4] dimension array
      [ do_nothing, left_thruster, main_thruster, right_thruster ] = np.squeeze(outputs.tolist())
      
      # !!! Transform the orientation of the right thruster based on location, and use that for the action???
      mirrored_right_thruster = transform_action(input, right_thruster)

      # Here we store the new action values, with the mirrored value replacing the right thruster.
      actions = np.array([[do_nothing, left_thruster, main_thruster, mirrored_right_thruster]])
        
      # !!! Here you need some logic to pick the action
      action = torch.argmax(torch.from_numpy(actions))

      # !!! I tried the default way of picking actions, which also produces an error...
      #action = torch.argmax(multitree.get_output_pt(input_sample))
        
      observation, reward, terminated, truncated, info =env.step(action.item())

      rewards.append(reward)
      output_sample = torch.from_numpy(observation.reshape((1,-1))).float()
      memory.push(input_sample, torch.tensor([[action.item()]]), output_sample, torch.tensor([reward]))
      if (terminated or truncated) and not ignore_done:
        break

  fitness = np.sum(rewards)
  
  return fitness, memory
