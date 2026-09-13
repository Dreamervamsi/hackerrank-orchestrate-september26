import sys
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))

print("Step 1: Importing packages...")
from code.data_manager import DataManager
from code.simulator import FinancialSimulator
from code.agent import StructuredFinancialAgent

print("Step 2: Initializing DataManager...")
dm = DataManager()

print("Step 3: Initializing Simulator...")
sim = FinancialSimulator(dm)

print("Step 4: Initializing Agent...")
ag = StructuredFinancialAgent(dm, sim)

print("Step 5: Processing request_26...")
t0 = time.time()
res = ag.process_request("request_26", "user_26")
print(f"Done in {time.time() - t0:.2f} seconds!")
print(res)
