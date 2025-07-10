"""
Feature extraction for the thesis implementation of the RTT feature
"""

import pandas as pd
from feature_extraction.extractors.base_extractor import BaseFeatureExtractor

class ThesisExtractor(BaseFeatureExtractor):
    def __init__(self, conn_df):
        super().__init__("Thesis implementation of RTT", conn_df)
        
        
    def _analyze_rtt_ratio(self, df, pkt_limit):
        """
        Analyzes network traffic data from a CSV string to compute RTT ratios using pandas.

        For each connection ('conn'), it identifies the first two "perfect round trips"
        and calculates a ratio between their time deltas.

        A "perfect round trip" is a sequence of three packets:
        1. Receiver -> Source
        2. Source -> Receiver
        3. Source -> Receiver
        """


        all_ratios = {}
        print("--- Starting RTT Ratio Analysis (using pandas) ---\n")

        # --- 2. Process each connection group ---
        for conn_id, group_df in df.groupby('conn'):
            # Convert the DataFrame group to a list of dictionaries for easier iteration
            packets = group_df.to_dict('records')

            if len(packets) < pkt_limit:
                print(f"[*] Connection '{conn_id}': Not enough packets to analyze. Skipping.\n")
                continue
            
            # Consider only the first pkt_limit number of packets
            group_df = group_df.head(pkt_limit)

            # --- 3. Identify original source and receiver from the first packet ---
            try:
                original_src_ip = packets[0]['src_ip']
                original_dst_ip = packets[0]['dst_ip']
            except (IndexError, KeyError) as e:
                print(f"[ERROR] Could not determine source/destination for '{conn_id}': {e}. Skipping.")
                continue

            print(f"[+] Processing Connection: '{conn_id}'")
            print(f"    - Original Source IP: {original_src_ip}")
            print(f"    - Original Receiver IP: {original_dst_ip}")

            # --- 4. Find all "perfect round trip" triplets ---
            found_triplets = []
            for i in range(len(packets) - 2):
                p1, p2, p3 = packets[i], packets[i+1], packets[i+2]
                original_src_ip = p1['dst_ip']
                original_dst_ip = p1['src_ip']

                # Check for the specific 3-packet pattern
                is_p1_correct = (p1['src_ip'] == original_dst_ip and p1['dst_ip'] == original_src_ip)
                is_p2_correct = (p2['src_ip'] == original_src_ip and p2['dst_ip'] == original_dst_ip)
                is_p3_correct = (p3['src_ip'] == original_src_ip and p3['dst_ip'] == original_dst_ip)

                if is_p1_correct and is_p2_correct and is_p3_correct:
                    found_triplets.append((p1, p2, p3))

            # --- 5. Calculate deltas and ratio if at least two triplets are found ---
            if len(found_triplets) >= 2:
                print(f"    - Found {len(found_triplets)} perfect round trip patterns.")
                
                first_triplet = found_triplets[0]
                second_triplet = found_triplets[1]

                # As per the request:
                # Delta 1: Time between 1st and 2nd packets of the FIRST triplet.
                # Delta 2: Time between 1st and 3rd packets of the SECOND triplet.
                try:
                    ts1_p1 = float(first_triplet[0]['ts_relative'])
                    ts1_p2 = float(first_triplet[1]['ts_relative'])
                    
                    ts2_p1 = float(second_triplet[0]['ts_relative'])
                    ts2_p3 = float(second_triplet[2]['ts_relative'])

                    delta1 = ts1_p2 - ts1_p1
                    delta2 = ts2_p3 - ts2_p1
                    
                    print("\n    --- First Triplet Found ---")
                    print(f"    1: {first_triplet[0]}")
                    print(f"    2: {first_triplet[1]}")
                    print(f"    3: {first_triplet[2]}")
                    print(f"    Delta 1 (ts[2] - ts[1]): {delta1:.10f}")

                    print("\n    --- Second Triplet Found ---")
                    print(f"    1: {second_triplet[0]}")
                    print(f"    2: {second_triplet[1]}")
                    print(f"    3: {second_triplet[2]}")
                    print(f"    Delta 2 (ts[3] - ts[1]): {delta2:.10f}")


                    if delta2 != 0:
                        ratio = delta1 / delta2
                        all_ratios[conn_id] = ratio
                        print(f"\n    => Ratio for '{conn_id}': {ratio:.6f}\n")
                    else:
                        print(f"\n    [!] Warning: Delta 2 is zero for '{conn_id}'. Cannot compute ratio. Skipping.\n")

                except (ValueError, KeyError) as e:
                    print(f"    [!] Error processing timestamps for '{conn_id}': {e}. Skipping.\n")

            else:
                print(f"    - Found only {len(found_triplets)} perfect round trip(s). Need at least 2 to compute a ratio. Skipping.\n")
        
        # --- 6. Calculate and print the final average ratio ---
        print("--- Analysis Complete ---")
        if all_ratios:
            return pd.DataFrame([{"conn": name, "rtt_ratio": value} for name, value in all_ratios.items()])
        else:
            print("\n[INFO] No ratios were calculated, so no average is available.")
            print("This may be because no connection had at least two 'perfect round trip' patterns in the provided data.")    
            return pd.DataFrame()
    
    def process_df(self, pkt_limit=20):
        return self._analyze_rtt_ratio(self.conn_df, pkt_limit)