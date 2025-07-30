#!/usr/bin/env python3
"""
Convert 3D velocity field arrays from HDF5 to binary format
compatible with the Phantom and sphNG smoothed particle 
hydrodynamics codes
"""

import h5py
import numpy as np
import struct

def write_fortran_unformatted(filename, data):
    """
    Write data in Fortran unformatted format with proper record markers.
    
    Parameters:
    filename: str - output filename
    data: numpy array - data to write
    """
    
    # Convert to 32-bit float (Fortran real(kind=4))
    data_f32 = data.astype(np.float32)
    
    # Ensure Fortran ordering for the nested loop pattern:
    # (((vgrid(i,j,k),i=1,nspace),j=1,nspace),k=1,nspace)
    data_fortran = np.asfortranarray(data_f32)
    
    # Calculate record length in bytes
    record_length = data_fortran.nbytes
    
    with open(filename, 'wb') as f:
        # Write opening record marker (4-byte integer)
        f.write(struct.pack('i', record_length))
        
        # Write the data
        f.write(data_fortran.tobytes())
        
        # Write closing record marker (4-byte integer)
        f.write(struct.pack('i', record_length))

def convert_velocity_field(input_file, output_files):
    """
    Convert HDF5 velocity field to binary files readable by Fortran.
    
    Parameters:
    input_file: str - path to HDF5 file containing vx, vy, vz datasets
    output_files: list - [cube_v1.dat, cube_v2.dat, cube_v3.dat] output file names
    """
    
    # Read HDF5 file
    print(f"Reading {input_file}...")
    with h5py.File(input_file, 'r') as h5file:
        # Read the velocity components
        vx = h5file['vx'][:]
        vy = h5file['vy'][:]
        vz = h5file['vz'][:]
        
        print(f"Array shape: {vx.shape}")
        print(f"Data type: {vx.dtype}")
    
    # List of velocity arrays and corresponding output files
    velocity_arrays = [vx, vy, vz]
    
    for i, (velocity, output_file) in enumerate(zip(velocity_arrays, output_files)):
        print(f"Converting component {i+1} to {output_file}...")
        
        # Write as Fortran unformatted binary file with record markers
        write_fortran_unformatted(output_file, velocity)
        
        # Check file size
        import os
        file_size = os.path.getsize(output_file)
        data_size = velocity.size * 4  # 4 bytes per float32
        expected_size = data_size + 8  # data + 2 record markers (4 bytes each)
        print(f"  Written {velocity.size} values + record markers ({file_size} bytes)")
        print(f"  Data: {data_size} bytes, Total with markers: {expected_size} bytes")

def main():
    # Input and output file names
    input_file = "velocity_field_seed0.h5"
    output_files = ["cube_v1.dat", "cube_v2.dat", "cube_v3.dat"]
    
    try:
        convert_velocity_field(input_file, output_files)
        print("\nConversion completed successfully!")
        print(f"Created files with Fortran unformatted record markers: {', '.join(output_files)}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 
