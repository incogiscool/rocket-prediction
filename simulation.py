import asyncio
import time
import utm


async def get_srad_packets_async(srad_data: dict, pos_dict: dict, interval: float = 1.0):
    """
    Async generator that yields SRAD packets and updates position dictionary.
    Overrides position data every second from SRAD packets.
    
    Args:
        srad_data: Parsed SRAD data dictionary
        pos_dict: Position dictionary to update with packet values
        interval: Time interval in seconds between packets (default: 1.0)
    """
    if not srad_data or 'packets' not in srad_data:
        return
    
    packets = srad_data.get('packets', [])
    
    for packet in packets:
        if isinstance(packet, dict) and 'samples' in packet:
            samples = packet.get('samples', [])
            
            if samples:
                # Get the latest sample
                latest_sample = samples[-1]

                easting, northing, _, _ = utm.from_latlon(latest_sample.get('lat', 0), latest_sample.get('lon', 0))
                # print(f"easting: {easting} northing {northing}")
                
                # Override position from SRAD data: x=lon, y=lat, z=alt
                pos_dict['x'] = easting
                pos_dict['y'] = northing
                pos_dict['z'] = latest_sample.get('alt', 0)
                
                # Calculate velocity based on last 2 samples
                if len(samples) >= 2:
                    prev_sample = samples[-2]
                    time_delta = latest_sample.get('time', 0) - prev_sample.get('time', 0)

                    prev_easting, prev_northing, _, _ = utm.from_latlon(prev_sample.get('lat', 0), prev_sample.get('lon', 0))
                    
                    if time_delta > 0:
                        pos_dict['velx'] = (easting - prev_easting) / time_delta
                        pos_dict['vely'] = (northing - prev_northing) / time_delta
                        pos_dict['velz'] = (latest_sample.get('alt', 0) - prev_sample.get('alt', 0)) / time_delta
                    else:
                        pos_dict['velx'] = 0
                        pos_dict['vely'] = 0
                        pos_dict['velz'] = 0
                else:
                    pos_dict['velx'] = 0
                    pos_dict['vely'] = 0
                    pos_dict['velz'] = 0
                
                # Override acceleration from latest sample
                pos_dict['accx'] = latest_sample.get('acc_x', 0)- 9.81
                pos_dict['accy'] = latest_sample.get('acc_y', 0)
                pos_dict['accz'] = latest_sample.get('acc_z', 0) 
                
                # Update the time of last update
                pos_dict['last_update_time'] = int(time.time() * 1000)
                
                print(f"\n[SRAD UPDATE] Packet ID: {packet.get('packet_id')} | Timestamp: {packet.get('timestamp')}")
                print(f"Position (easting, northing, alt): ({pos_dict['x']:.6f}m, {pos_dict['y']:.6f}m, {pos_dict['z']:.6f})m")
                print(f"Velocity (x, y, z): ({pos_dict['velx']:.6f}, {pos_dict['vely']:.6f}, {pos_dict['velz']:.6f})")
                print(f"Acceleration (x, y, z): ({pos_dict['accx']:.6f}, {pos_dict['accy']:.6f}, {pos_dict['accz']:.6f})\n")
        
        # Wait for interval before next packet
        await asyncio.sleep(interval)
