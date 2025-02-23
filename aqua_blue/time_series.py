from typing import IO, Union
from pathlib import Path

from dataclasses import dataclass
import numpy as np
from datetime import datetime, timedelta
import pandas as pd

from numpy.typing import NDArray


@dataclass
class TimeSeries:

    dependent_variable: NDArray
    times: NDArray

    def __post_init__(self):
        # Convert integer-based time indices to datetime starting from a fixed date
        start_date = datetime(1970, 1, 1)
        self.times = np.array([start_date + timedelta(days=t) for t in self.times])
        
        timesteps = np.diff(self.times.astype('datetime64[D]').astype(float))

        if not np.isclose(np.std(timesteps), 0.0):
            raise ValueError("TimeSeries.times must be uniformly spaced")
        
        if np.isclose(np.mean(timesteps), 0.0):
            raise ValueError("TimeSeries.times must have a timestep greater than zero")

    def save(self, file: IO, header="", delimiter=","):
        np.savetxt(
            file,
            np.vstack((self.times.astype('datetime64[D]').astype(float), self.dependent_variable.T)).T
            delimiter=delimiter,
            header=header,
            comments=""
        )

    @property
    def num_dims(self) -> int:

        return self.dependent_variable.shape[1]

    @classmethod
    def from_csv(cls, fp: Union[IO, str, Path], time_index: int = 0):

        data = np.loadtxt(fp, delimter=",")
        return cls(
            dependent_variable=np.delete(data, obj=time_index, axis=1),
            times=data[:,time_index]
        )

    @property
    def timestep(self) -> float:

        return (self.times[1] - self.times[0]).days # returns timestep in days

    def __eq__(self, other) -> bool:

        return bool(np.all(self.times == other.times) and np.all(
            np.isclose(self.dependent_variable, other.dependent_variable)
        ))
