# System_Builder
This is a small system builder for openmm systems, it is used to pack simple boxes and generating forcefields for openmm.

It exists while openff packages are not pip installable. it contains a main script which can be used to pack an parameterize simple systems and export them so that they can be read in openmm scripts. This decouples the need to use conda with openmm while needing openff.