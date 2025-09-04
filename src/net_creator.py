import random
import string

def generate_epanet_input(
    filename="complex_network.inp",
    num_reservoirs=2,
    num_tanks=8,
    num_junctions=500,
    num_pipes=200,
    num_prvs=40,
    num_pumps=10,
    time_patterns=24
):
    def rand_id(prefix, i):
        return f"{prefix}{i+1}"

    def rand_float(a, b):
        return round(random.uniform(a, b), 2)

    with open(filename, "w") as f:
        f.write("[TITLE]\n;; Example Complex Water Network\n\n")

        # Junctions
        f.write("[JUNCTIONS]\n;;ID Elevation Demand Pattern\n")
        for i in range(num_junctions):
            jid = rand_id("J", i)
            elev = rand_float(50, 150)
            demand = rand_float(0.1, 2.0)
            f.write(f"{jid:<10} {elev:<10} {demand:<10} DPat\n")
        f.write("\n")

        # Reservoirs
        f.write("[RESERVOIRS]\n;;ID Head\n")
        for i in range(num_reservoirs):
            rid = rand_id("R", i)
            head = rand_float(120, 200)
            f.write(f"{rid:<10} {head}\n")
        f.write("\n")

        # Tanks
        f.write("[TANKS]\n;;ID Elevation InitLevel MinLevel MaxLevel Diameter MinVol\n")
        for i in range(num_tanks):
            tid = rand_id("T", i)
            elev = rand_float(30, 100)
            init = rand_float(5, 15)
            f.write(f"{tid:<10} {elev:<10} {init:<10} {elev-5:<10} {elev+20:<10} {rand_float(10, 20):<10} {rand_float(50, 200)}\n")
        f.write("\n")

        # Pipes
        f.write("[PIPES]\n;;ID Node1 Node2 Length Diameter Roughness MinorLoss\n")
        for i in range(num_pipes):
            pid = rand_id("P", i)
            n1 = rand_id("J", random.randrange(num_junctions))
            n2 = rand_id("J", random.randrange(num_junctions))
            length = random.randint(100, 1000)
            diameter = random.choice([100, 150, 200, 250])
            rough = rand_float(80, 120)
            f.write(f"{pid:<10} {n1:<5} {n2:<5} {length:<8} {diameter:<8} {rough:<10} 0\n")
        f.write("\n")

        # PRVs
        f.write("[VALVES]\n;;ID Node1 Node2 Type Setting\n")
        for i in range(num_prvs):
            vid = rand_id("V", i)
            n1 = rand_id("J", random.randrange(num_junctions))
            n2 = rand_id("J", random.randrange(num_junctions))
            f.write(f"{vid:<10} {n1:<5} {n2:<5} PRV 60\n")
        f.write("\n")

        # Pumps
        f.write("[PUMPS]\n;;ID Node1 Node2 Curve\n")
        for i in range(num_pumps):
            pu = rand_id("PU", i)
            n1 = rand_id("R", random.randrange(num_reservoirs))
            n2 = rand_id("J", random.randrange(num_junctions))
            crv = rand_id("C", i)
            f.write(f"{pu:<10} {n1:<5} {n2:<5} {crv}\n")
        f.write("\n")

        # Demand Patterns
        f.write("[PATTERNS]\n;;ID Factors...\n")
        f.write("DPat      " + " ".join(f"{rand_float(0.5, 1.5):<6}" for _ in range(time_patterns)) + "\n\n")

        # Curves
        f.write("[CURVES]\n;;ID X-Value Y-Value\n")
        for i in range(num_pumps):
            cid = rand_id("C", i)
            for q in [0, 100, 200, 300]:
                head = round(60 - 0.05*q + random.uniform(-2,2),2)
                f.write(f"{cid:<10} {q:<8} {head}\n")
        f.write("\n")

        # Controls
        f.write("[CONTROLS]\n;; Example: Turn on pump PU1 at 6:00\nTIME            06:00   OPEN    PU1\n\n")

        # Options, Report, Coordinates, Vertices
        f.write("[OPTIONS]\nUNITS             LPS\nHEADLOSS          H-W\n\n[REPORT]\nSUMMARY          YES\n\n[COORDINATES]\n;;Node X Y\n")
        for i in range(num_junctions):
            jid = rand_id("J", i)
            f.write(f"{jid:<10} {random.randint(0,1000):<6} {random.randint(0,1000)}\n")
        f.write("\n[VERTICES]\n;;Pipe X Y\n")
        for i in range(num_pipes):
            pid = rand_id("P", i)
            f.write(f"{pid:<10} {random.randint(0,1000):<6} {random.randint(0,1000)}\n")

    print(f"Generated {filename}")

# Example usage
generate_epanet_input()
