from b_agent_mmwave import RadarAgent

agent = RadarAgent()

while True:

    user_input = input("\nUser: ")

    if user_input == "exit":
        break

    result = agent.run(user_input)

    print("Agent:", type(result))
