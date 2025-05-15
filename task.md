minimal-agents-project/
│
├── README.md
├── requirements.txt
├── main.py                        # Entry point for the simulation
│
├── config/
│   └── settings.py               # Simulation parameters (speed, size, agent count, etc.)
│
├── core/                         # Core logic for environment and game loop
│   ├── environment.py            # World setup, grid/map, obstacles, resources
│   └── simulator.py              # Main simulation loop (PyGame rendering, ticking)
│
├── agents/                       # Agent definitions and internal models
│   ├── base_agent.py             # Abstract Agent class with sensors/motors
│   ├── agent_logic.py            # Agent decision rules (e.g., Pyke/PyDatalog)
│   ├── simple_agent.py           # A minimal reactive Braitenberg-style agent
│   └── stress_agent.py           # A more complex rule-based or stateful agent
│
├── assets/                       # Images/sprites for agents, map, etc. (optional)
│   ├── agent.png
│   └── world_bg.png
│
├── utils/                        # Utilities, helper functions
│   ├── logger.py                 # Logging agent behavior / stress levels etc.
│   └── math_utils.py             # Vector math, collision checks, etc.
│
└── docs/                         # Reports, diagrams, and reference material
    ├── architecture.png          # Diagram of components
    └── project_notes.md          # Design notes and research insights