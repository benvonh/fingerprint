import os
import math
import shutil
import cv2 as cv
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
from tkinter import filedialog
from utils import * # custom
from typing import Any
import matplotlib
# Use a non-interactive backend to ensure no pop-up windows are created
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

class App(tk.Tk):
    THRESHOLD = 0.63
    DATABASE_DIR = './SimpleFingerprintRecognitionExample/database'

    def __init__(self):
        super().__init__()
        super().title("Fingerprint Recognition GUI")  
        super().geometry("800x200")
        
        # Enrol fingerprint
        self.enrol_label = tk.Label(self, text="No selected file for enrolment", justify='left')
        self.enrol_button = tk.Button(self, text="Enrol Fingerprint", command=self.enrol_fingerprint)
        self.enrol_label.grid(column=1, row=0, padx=10, pady=10, sticky='w')
        self.enrol_button.grid(column=0, row=0, padx=10, pady=10, sticky='w')

        # Compare fingerprint
        self.compare_label = tk.Label(self, text="No selected file for comparison", justify='left')
        self.compare_button = tk.Button(self, text="Compare Fingerprint", command=self.compare_fingerprint)
        self.compare_label.grid(column=1, row=1, padx=10, pady=10, sticky='w')
        self.compare_button.grid(column=0, row=1, padx=10, pady=10, sticky='w')

        # Auto-evaluate threshold
        self.eval_label = tk.Label(self, text="No auto-evaluation", justify='left')
        self.eval_button = tk.Button(self, text="Auto-evaluate Threshold", command=self.evaluate_threshold)
        self.eval_label.grid(column=1, row=2, padx=10, pady=10, sticky='w')
        self.eval_button.grid(column=0, row=2, padx=10, pady=10, sticky='w')

        # Fingerprint comparison result
        self.compare_result_label = tk.Label(self, text="No comparison result", justify='left')
        self.compare_result_label.grid(column=0, row=3, padx=10, pady=10, sticky='w', columnspan=2)


    @staticmethod
    def file_dialog() -> str:
        return filedialog.askopenfilename(
            initialdir="./DB1_B",
            title="Select Fingerprint",
            filetypes=(
                ("Image", "*.tif"),
                ("All files", "*.*")
            ))


    @staticmethod
    def analyze_fingerprint(file_path: str) -> Any:
        fingerprint = cv.imread(file_path, cv.IMREAD_GRAYSCALE)

        gx, gy = cv.Sobel(fingerprint, cv.CV_32F, 1, 0), cv.Sobel(fingerprint, cv.CV_32F, 0, 1)

        gx2, gy2 = gx**2, gy**2
        gm = np.sqrt(gx2 + gy2)

        sum_gm = cv.boxFilter(gm, -1, (25, 25), normalize = False)

        thr = sum_gm.max() * 0.2
        mask = cv.threshold(sum_gm, thr, 255, cv.THRESH_BINARY)[1].astype(np.uint8)

        W = (23, 23)
        gxx = cv.boxFilter(gx2, -1, W, normalize = False)
        gyy = cv.boxFilter(gy2, -1, W, normalize = False)
        gxy = cv.boxFilter(gx * gy, -1, W, normalize = False)
        gxx_gyy = gxx - gyy
        gxy2 = 2 * gxy

        orientations = (cv.phase(gxx_gyy, -gxy2) + np.pi) / 2 
        sum_gxx_gyy = gxx + gyy

        region = fingerprint[10:90,80:130]

        smoothed = cv.blur(region, (5,5), -1)
        xs = np.sum(smoothed, 1) 

        local_maxima = np.nonzero(np.r_[False, xs[1:] > xs[:-1]] & np.r_[xs[:-1] >= xs[1:], False])[0]

        distances = local_maxima[1:] - local_maxima[:-1]

        ridge_period = np.average(distances)
        
        or_count = 8
        gabor_bank = [gabor_kernel(ridge_period, o) for o in np.arange(0, np.pi, np.pi/or_count)]
        
        nf = 255-fingerprint
        all_filtered = np.array([cv.filter2D(nf, cv.CV_32F, f) for f in gabor_bank])
        
        y_coords, x_coords = np.indices(fingerprint.shape)
        
        orientation_idx = np.round(((orientations % np.pi) / np.pi) * or_count).astype(np.int32) % or_count
        
        filtered = all_filtered[orientation_idx, y_coords, x_coords]
        
        enhanced = mask & np.clip(filtered, 0, 255).astype(np.uint8)
        
        _, ridge_lines = cv.threshold(enhanced, 32, 255, cv.THRESH_BINARY)
        
        skeleton = cv.ximgproc.thinning(ridge_lines, thinningType = cv.ximgproc.THINNING_GUOHALL)
        
        def compute_crossing_number(values):
            return np.count_nonzero(values < np.roll(values, -1))
        
        cn_filter = np.array([[  1,  2,  4],
                            [128,  0,  8],
                            [ 64, 32, 16]
                            ])
        
        all_8_neighborhoods = [np.array([int(d) for d in f'{x:08b}'])[::-1] for x in range(256)]
        cn_lut = np.array([compute_crossing_number(x) for x in all_8_neighborhoods]).astype(np.uint8)
        
        skeleton01 = np.where(skeleton!=0, 1, 0).astype(np.uint8)
        
        neighborhood_values = cv.filter2D(skeleton01, -1, cn_filter, borderType = cv.BORDER_CONSTANT)
        
        cn = cv.LUT(neighborhood_values, cn_lut)
        
        cn[skeleton==0] = 0
        
        minutiae = [(x,y,cn[y,x]==1) for y, x in zip(*np.where(np.isin(cn, [1,3])))]
        
        mask_distance = cv.distanceTransform(cv.copyMakeBorder(mask, 1, 1, 1, 1, cv.BORDER_CONSTANT), cv.DIST_C, 3)[1:-1,1:-1]

        filtered_minutiae = list(filter(lambda m: mask_distance[m[1], m[0]]>10, minutiae))
        
        def compute_next_ridge_following_directions(previous_direction, values):    
            next_positions = np.argwhere(values!=0).ravel().tolist()
            if len(next_positions) > 0 and previous_direction != 8:
                next_positions.sort(key = lambda d: 4 - abs(abs(d - previous_direction) - 4))
                if next_positions[-1] == (previous_direction + 4) % 8: 
                    next_positions = next_positions[:-1] 
            return next_positions
        
        r2 = 2**0.5 
        
        xy_steps = [(-1,-1,r2),( 0,-1,1),( 1,-1,r2),( 1, 0,1),( 1, 1,r2),( 0, 1,1),(-1, 1,r2),(-1, 0,1)]
        
        nd_lut = [[compute_next_ridge_following_directions(pd, x) for pd in range(9)] for x in all_8_neighborhoods]
        
        def follow_ridge_and_compute_angle(x, y, d = 8):
            px, py = x, y
            length = 0.0
            while length < 20: 
                next_directions = nd_lut[neighborhood_values[py,px]][d]
                if len(next_directions) == 0:
                    break
                
                if (any(cn[py + xy_steps[nd][1], px + xy_steps[nd][0]] != 2 for nd in next_directions)):
                    break 
                
                d = next_directions[0]
                ox, oy, l = xy_steps[d]
                px += ox ; py += oy ; length += l
            
            return math.atan2(-py+y, px-x) if length >= 10 else None
        
        valid_minutiae = []
        for x, y, term in filtered_minutiae:
            d = None
            if term: 
                d = follow_ridge_and_compute_angle(x, y)
            else: 
                dirs = nd_lut[neighborhood_values[y,x]][8] 
                if len(dirs)==3: 
                    angles = [follow_ridge_and_compute_angle(x+xy_steps[d][0], y+xy_steps[d][1], d) for d in dirs]
                    if all(a is not None for a in angles):
                        a1, a2 = min(((angles[i], angles[(i+1)%3]) for i in range(3)), key=lambda t: angle_abs_difference(t[0], t[1]))
                        d = angle_mean(a1, a2)                
            if d is not None:
                valid_minutiae.append( (x, y, term, d) )

        mcc_radius = 70
        mcc_size = 16

        g = 2 * mcc_radius / mcc_size
        x = np.arange(mcc_size)*g - (mcc_size/2)*g + g/2
        y = x[..., np.newaxis]
        iy, ix = np.nonzero(x**2 + y**2 <= mcc_radius**2)
        ref_cell_coords = np.column_stack((x[ix], x[iy]))

        mcc_sigma_s = 7.0
        mcc_tau_psi = 400.0
        mcc_mu_psi = 1e-2

        def Gs(t_sqr):
            """"Gaussian function with zero mean and mcc_sigma_s standard deviation, see eq. (7) in MCC paper"""
            return np.exp(-0.5 * t_sqr / (mcc_sigma_s**2)) / (math.tau**0.5 * mcc_sigma_s)

        def Psi(v):
            """"Sigmoid function that limits the contribution of dense minutiae clusters, see eq. (4)-(5) in MCC paper"""
            return 1. / (1. + np.exp(-mcc_tau_psi * (v - mcc_mu_psi)))

        xyd = np.array([(x,y,d) for x,y,_,d in valid_minutiae]) 
        
        d_cos, d_sin = np.cos(xyd[:,2]).reshape((-1,1,1)), np.sin(xyd[:,2]).reshape((-1,1,1))
        rot = np.block([[d_cos, d_sin], [-d_sin, d_cos]])
        
        xy = xyd[:,:2]
        
        cell_coords = np.transpose(rot@ref_cell_coords.T + xy[:,:,np.newaxis],[0,2,1])
        
        dists = np.sum((cell_coords[:,:,np.newaxis,:] - xy)**2, -1)
        
        cs = Gs(dists)
        diag_indices = np.arange(cs.shape[0])
        cs[diag_indices,:,diag_indices] = 0 
        
        local_structures = Psi(np.sum(cs, -1))
        return local_structures
    
    
    @staticmethod
    def get_score(ls1: Any, ls2: Any) -> float:
        dists = np.linalg.norm(ls1[:,np.newaxis,:] - ls2, axis = -1)
        dists /= np.linalg.norm(ls1, axis = 1)[:,np.newaxis] + np.linalg.norm(ls2, axis = 1)
        num_p = int(min(len(ls1), len(ls2)))
        pairs = np.unravel_index(np.argpartition(dists, num_p, None)[:num_p], dists.shape)
        return 1 - np.mean(dists[pairs[0], pairs[1]])


    def enrol_fingerprint(self) -> None:
        file_path = self.file_dialog()

        if file_path == '':
            return

        ls = self.analyze_fingerprint(file_path)

        save_path = os.path.join(self.DATABASE_DIR, os.path.splitext(os.path.basename(file_path))[0] + ".npz")
        np.savez(save_path, ls=ls)

        self.enrol_label.config(text=f"Enrolled fingerprint in database at {save_path}")


    def compare_fingerprint(self) -> None:
        file_path = self.file_dialog()

        if file_path is None:
            return

        ls1 = self.analyze_fingerprint(file_path)
        database = {fingerprint: 0.0 for fingerprint in os.listdir(self.DATABASE_DIR)}

        for ofn in database.keys():
            ofn_path = os.path.join(self.DATABASE_DIR, ofn)
            ls2 = np.load(ofn_path, allow_pickle=True)['ls']
            database[ofn] = self.get_score(ls1, ls2)

        self.compare_label.config(text=f"Compared fingerprint against database for {file_path}")

        compare_result_text = f"Matched Fingerprints (threshold={self.THRESHOLD}):\n"\
                               "--------------------------------------------------\n"\
                               " FINGERPRINT : SCORE\n"\
                               "--------------------------------------------------\n"

        for fn, score in database.items():
            if score > self.THRESHOLD:
                compare_result_text += f" {fn} : {score:.2f}\n"

        self.compare_result_label.config(text=compare_result_text)


    def evaluate_threshold(self) -> None:
        print("Analyzing fingerprints...")
        fingerprint_paths = [os.path.basename(fp) for fp in os.listdir('./DB1_B')]
        fingerprint_paths.sort()
        local_structures = [App.analyze_fingerprint(os.path.join('./DB1_B', fp)) for fp in fingerprint_paths]
        fingerprints = list()

        # Format: "XXX_Y" (fingerprint ID, fingerprint V)
        # Only care about ID
        for p in fingerprint_paths:
            fingerprints.append(int(os.path.splitext(p)[0].split('_')[0]))

        scores = list()
        tpr_list = list()
        fpr_list = list()

        print("Scoring finerprints...")
        for th in range(0, 100, 1):
            for i in range(0, len(local_structures) - 1):
                ls1 = local_structures[i]
                for j in range(i+1, len(local_structures)):
                    ls2 = local_structures[j]
                    scores.append((App.get_score(ls1, ls2), i, j))
                
            print(f"Progress: {th+1}%", end='\r')

        print("Rating fingerprints...")
        for th in range(0, 100, 1):
            tn = 0
            fp = 0
            fn = 0
            tp = 0
            threshold = th / 100.0
            for sc in scores:
                score = sc[0]
                i = sc[1]
                j = sc[2]
                if score > threshold:
                    if fingerprints[i] == fingerprints[j]:
                        tp += 1
                    else:
                        fp += 1
                else:
                    if fingerprints[i] != fingerprints[j]:
                        tn += 1
                    else:
                        fn += 1
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            tpr_list.append(tpr)
            fpr_list.append(fpr)
            print(f"Threshold: {th}%, TPR: {int(100*tpr)}%, FPR: {int(100*fpr)}%")

        self.eval_label.config(text=f"Auto-evaluation complete")
        plot_and_export_roc_curve(fpr_list, tpr_list)


def plot_and_export_roc_curve(fpr, tpr, output_file='roc_curve.png'):
    """
    Computes the ROC curve and AUC from the true labels and predicted scores.
    Plots the ROC curve and exports it to the specified file without displaying a pop-up window.

    Parameters:
    - y_true: array-like of shape (n_samples,)
        True binary labels. 
    - y_scores: array-like of shape (n_samples,)
        Target scores, can either be probability estimates of the positive class,
        confidence values, or non-thresholded measure of decisions.
    - output_file: str, default='roc_curve.png'
        The path (with filename and extension) to save the ROC curve plot.
    """
    roc_auc = auc(fpr, tpr)
    
    # Create a new figure
    plt.figure()
    
    # Plot the ROC curve
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label='ROC curve (AUC = {:.2f})'.format(roc_auc))
    
    # Plot the diagonal line representing a random classifier
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    # Set the limits for the axes
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    # Add labels and title
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    
    # Display a legend in the lower right corner
    plt.legend(loc='lower right')
    
    # Save the figure to a file (this prevents any pop-up window from appearing)
    plt.savefig(output_file, format='png')
    
    # Close the figure to free up memory and prevent display
    plt.close()


if __name__ == "__main__":
    app = App()
    app.mainloop()