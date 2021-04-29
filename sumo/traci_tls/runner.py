#!/usr/bin/python3

from __future__ import absolute_import, print_function
import os
import sys
import optparse
import random

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary # noqa
import traci # noqa


def generate_routefile():
    random.seed(42)  # make test reproducible
    N = 3600  # Number of timesteps
    # demand per second from different directions
    pWE = 1. / 10
    pEW = 1. / 11
    pNS = 1. / 30
    with open("data/traci_tls.rou.xml", "w") as routes:
        print('<routes>', file=routes)
        print('    <vType id="typeWE" accel="0.8" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="16.67" guiShape="passenger" />', file=routes)
        print('    <vType id="typeNS" accel="0.8" decel="4.5" sigma="0.5" length="7" minGap="3" maxSpeed="25" guiShape="bus" />', file=routes)
        print('', file=routes)
        print('    <route id="right" edges="51o 1i 2o 52i" />', file=routes)
        print('    <route id="left" edges="52o 2i 1o 51i" />', file=routes)
        print('    <route id="down" edges="54o 4i 3o 53i" />', file=routes)
        print('', file=routes)

        vehNr = 0
        for i in range(N):
            if random.uniform(0, 1) < pWE:
                print('    <vehicle id="right_%i" type="typeWE" route="right" depart="%i" />' % (vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pEW:
                print('    <vehicle id="left_%i" type="typeWE" route="left" depart="%i" />' % (vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pNS:
                print('    <vehicle id="down_%i" type="typeNS" route="down" depart="%i" color="1,0,0" />' % (vehNr, i), file=routes)
                vehNr += 1
        print('</routes>', file=routes)


def run():
    """Execute the Traci control loop"""
    step = 0
    traci.trafficlight.setPhase("0", 2)                                # we start with phase 2 where EW has green
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        if traci.trafficlight.getPhase("0") == 2:
            if traci.inductionloop.getLastStepVehicleNumber("0") > 0:  # we are not already switching
                traci.trafficlight.setPhase("0", 3)                    # there is a vehicle from the north, switch
            else:
                traci.trafficlight.setPhase("0", 2)                    # otherwise try to keep green for EW
        step += 1
    traci.close()
    sys.stdout.flush()


def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option('--nogui', action="store_true", default=False, help="run the commandline version of sumo")
    options, args = optParser.parse_args()
    return options


# this is the main entry point of this script
if __name__ == "__main__":
    options = get_options()

    # this script has been called from the command line. It will start sumo as a server, then connect and run.
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    # first, generate the route file for this simulation
    generate_routefile()

    # this is the normal way of using traci
    # sumo is started as a subprocess and then the python script connects and runs
    traci.start([sumoBinary, "-c", "data/traci_tls.sumocfg", "--tripinfo-output", "data/tripinfo.xml"])
    run()
