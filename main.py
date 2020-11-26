import os

from train.GTurbo.fuel import GFuel

opening_folder = './data/instances_ss_confirmed'
opening_file = 'openings_instances_avg'

file = os.path.join(opening_folder, opening_file)

gFuel = GFuel(file_name=file)

print(gFuel.get_fuel())


