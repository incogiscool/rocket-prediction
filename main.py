import time
import asyncio
import formulas
import util
import simulation
import filters

srad_data = util.parse_json("./srad_data.json")

pos = {
    "x": 0,
    "y": 0,
    "z": 0,
    "velx": 0,
    "vely": 0,
    "velz": 0,
    "accx": 0,
    "accy": 0,
    "accz": 0,
    "last_update_time": int(time.time() * 1000)
}

async def main():
    """Main async function that runs both the packet reader and position calculator."""
    # Start the packet reading task
    packet_task = asyncio.create_task(simulation.get_srad_packets_async(srad_data, pos, interval=1.0))
    
    # Run the position calculation loop
    try:
        while True:
            # Calculate elapsed time since last SRAD update
            curr_time = int(time.time() * 1000)
            time_elapsed = ((curr_time - pos['last_update_time']) / 1000) + 2
            ALPHA = 0.35
            
            # Update position using constant acceleration equation for each dimension. Flipped x and z because accelerometer seems to be mounted differently
            pos["x"] = filters.lowpass(formulas.const_acc_eq(pos["velz"], time_elapsed, pos["accz"], pos['x']), pos['x'], ALPHA)
            pos["y"] = filters.lowpass(formulas.const_acc_eq(pos["vely"], time_elapsed, pos["accy"], pos['y']), pos['y'], ALPHA)
            pos["z"] = filters.lowpass(formulas.const_acc_eq(pos["velx"], time_elapsed, pos["accx"], pos['z']), pos['z'], ALPHA)

            # pos["x"] = formulas.const_acc_eq(pos["velx"], time_elapsed, pos["accx"], pos['x'])
            # pos["y"] = formulas.const_acc_eq(pos["vely"], time_elapsed, pos["accy"], pos['y'])
            # pos["z"] = formulas.const_acc_eq(pos["velz"], time_elapsed, pos["accz"], pos['z'])
            
            print(f"Time since update: {time_elapsed:.3f}s | Position: ({pos['x']:.6f}, {pos['y']:.6f}, {pos['z']:.6f})")
            
            await asyncio.sleep(0.1)
    except KeyboardInterrupt:
        packet_task.cancel()


# Run the main async function
asyncio.run(main())
