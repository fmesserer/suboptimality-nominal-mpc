from dataclasses import dataclass
import pickle
import numpy as np

@dataclass
class ResultsValFunc():

    x_vals: np.ndarray
    sig_vals: np.ndarray
    u0_opt: np.ndarray
    u0_sub: np.ndarray
    val_opt: np.ndarray
    val_sub: np.ndarray

    def get_data(self):
        return (
            self.x_vals,
            self.sig_vals,
            self.u0_opt,
            self.u0_sub,
            self.val_opt,
            self.val_sub,
        )

    def save(self, filename: str):
        print("Saving data to %s" % filename)
        data = self.get_data()
        with open(filename, "wb") as file:
            pickle.dump(data, file)

    @staticmethod
    def load(filename: str):
        print("Loading data from %s" % filename)
        with open(filename, "rb") as file:
            (
                x_val,
                sig_vals,
                u0_opt,
                u0_sub,
                val_opt,
                val_sub,
            ) = pickle.load(file)
        results = ResultsValFunc(
                    x_vals=x_val,
                    sig_vals=sig_vals,
                    u0_opt=u0_opt,
                    u0_sub=u0_sub,
                    val_opt=val_opt,
                    val_sub=val_sub,
                    )
        return results
    