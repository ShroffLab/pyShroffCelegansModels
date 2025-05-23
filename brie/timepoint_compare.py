import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from skimage import io,metrics
import csv
import re

def check_case(string):

    if string.isdigit() == True:
        pass
    elif string.startswith(('ca','cp','d','ep','ea')):
        string = string[:1].upper() + string[1:].lower()
        return string
    elif string.startswith(('ab','ms','dd')):
        string = string[:2].upper() + string[2:].lower()
        return string
    else:
        return string

def check_prefix_suffix(string):

    if string.lower().startswith('hyp') and len(string) >= 7:
        i = string.find('_')
        string = string.replace(string[:i+1],'')
        return string

    elif re.search(r'_hyp\w*$',string):
        match = re.search(r'_hyp\w*$',string)
        string = string.replace(match.group(),'')
        return string

    elif re.search(r'_death\w*$',string):
        match = re.search(r'_death\w*$',string)
        string = string.replace(match.group(),'')
        return string
    else:
        return string

def check_len(string):

    if len(string) <= 3:
        return True
    else:
        return False

def is_digit(string):

    if str(string).isdigit() != True:
        return True
    else:
        return False

def lower(string):
    string = str(string)
    string = string.lower()
    return string

def normalize(val,_min,_max):
    norm = (val-_min)/(_max-_min)
    return norm

class Loader:

    def __init__(self,df,key_df,embryo_summary):

        # Lookup libraries
        self.summary = df
        self.embryo_summary = embryo_summary
        self.strains = list(self.summary.groupby('lineage').groups.keys())
        self.strain_index = []
        self.embryo_index = []

        # File information
        self.filepaths = []
        self.embryo_filepaths = []
        self.images = []
        self.embryo_images = []

        # Lineage Dataframes
        self.data = {}
        self.seam_cells = {}
        self.normalized = {}
        self.norm_seam_cells = {}
        self.norm_embryo_seam_cells = {}
        self.lineage_key_df = key_df.map(lower)

        # Embryo Dataframes
        self.embryo = None
        self.embryo_seam_cells = {}
        self.embryo_raw = None
        self.norm_embryo = None
        self.norm_embyro_raw = None

        # Aggregation Information
        self.cellkeys = {}
        self.cellnames = {}
        self.data_by_lineage = {}
        self.all_coords = None
        self.all_coords_norm = None
        self.aggregate_data = {}
        self.aggregate_data_normalized = {}

        # Statistics
        self.avg = {}
        self.avg_normalized = {}
        self.stdev = {}
        self.stdev_normalized = {}

        ##### INITIALIZE EMBRYO COORDINATE DATA ######
        #Read in embryo coordinate information
        embryo_data = {}
        normalized_embryo_data = {}
        for r in self.embryo_summary.iterrows():
            
            file = r[1].iloc[0]
            t0 = r[1].iloc[1]

            annotations = file + f"\\Decon_reg_{t0}\\Decon_reg_{t0}_results\\straightened_annotations\\straightened_annotations.csv"
            self.embryo_filepaths.append(f"{annotations}")

            img = file + f"\\Decon_reg_{t0}\\Decon_reg_{t0}_results\\output_images\\Decon_reg_{t0}_straight.tif"
            self.embryo_images.append(f"{img}")

            seam = file + f"\\Decon_reg_{t0}\\Decon_reg_{t0}_results\\straightened_lattice\\straightened_lattice.csv"
            seam = pd.read_csv(f"{seam}",index_col='name')[['x_voxels','y_voxels','z_voxels']]
            seam = seam.drop([x for x in seam.index if x.startswith('a')])
            self.embryo_seam_cells.setdefault(f"Embryo_{t0-1}",seam)

            img = io.imread(img)
            x_bounds,y_bounds=img[0].shape
            z_bounds = len(img)

            data = pd.read_csv(f"{annotations}")[['name','x_voxels','y_voxels','z_voxels']].dropna(axis=1,thresh=1).dropna(axis=0,thresh=1)
            data['name'] = list(map(lambda x: x.replace(' ',''),data['name']))
            embryo_data.setdefault(f"Embryo_{t0-1}",data)
            self.embryo_index.append(f"Embryo_{t0-1}")

            norm = data.copy()
            norm[['x_voxels','y_voxels','z_voxels']] = norm[['x_voxels','y_voxels','z_voxels']].astype(np.float64)
            norm['x_voxels'] = norm['x_voxels'].apply(normalize,_min=0,_max=int(x_bounds))
            norm['y_voxels'] = norm['y_voxels'].apply(normalize,_min=0,_max=int(y_bounds))
            norm['z_voxels'] = norm['z_voxels'].apply(normalize,_min=0,_max=int(z_bounds))

            normalized_embryo_data.setdefault(f"Embryo_{t0-1}",norm)

            seam_n = seam.copy()
            seam_n[['x_voxels','y_voxels','z_voxels']] = seam_n[['x_voxels','y_voxels','z_voxels']].astype(np.float64).copy()
            seam_n['x_voxels'] = seam_n['x_voxels'].apply(normalize,_min=0,_max=int(x_bounds))
            seam_n['y_voxels'] = seam_n['y_voxels'].apply(normalize,_min=0,_max=int(y_bounds))
            seam_n['z_voxels'] = seam_n['z_voxels'].apply(normalize,_min=0,_max=int(z_bounds))
            self.norm_embryo_seam_cells.setdefault(f"Embryo_{t0-1}",seam_n)

        # Collect all embryo data to one dataframe
        embryo_0 = embryo_data["Embryo_0"].set_index('name')
        embryo_1 = embryo_data["Embryo_1"].set_index('name')
        embryo_2 = embryo_data["Embryo_2"].set_index('name')

        self.embryo_raw = pd.concat(objs=[embryo_0,embryo_1,embryo_2],axis=1,join='inner')
        embryo_lineage_names = list(map(lambda x: x.replace(' ',''),self.embryo_raw.index))

        # Average the absolute coordinates across embryos
        x = self.embryo_raw['x_voxels'].aggregate(['mean',lambda x: x.std(ddof=0)],axis=1)
        y = self.embryo_raw['y_voxels'].aggregate(['mean',lambda x: x.std(ddof=0)],axis=1)
        z = self.embryo_raw['z_voxels'].aggregate(['mean',lambda x: x.std(ddof=0)],axis=1)

        avg = np.hstack((
            np.array(x['mean']).reshape((len(x),1)),
            np.array(y['mean']).reshape((len(y),1)),
            np.array(z['mean']).reshape((len(z),1)),
            ))
        avg = pd.DataFrame(data=avg,index=embryo_lineage_names,columns=['x mean','y mean','z mean']).sort_index()
        self.embryo = avg.round(0)

        # Collect all normalized embryo data to one dataframe
        embryo_0 = normalized_embryo_data["Embryo_0"].set_index('name')
        embryo_1 = normalized_embryo_data["Embryo_1"].set_index('name')
        embryo_2 = normalized_embryo_data["Embryo_2"].set_index('name')
        
        self.norm_embryo_raw = pd.concat(objs=[embryo_0,embryo_1,embryo_2],axis=1,join='inner')

        # Average the normalized coordinates across embryos   
        x = self.norm_embryo_raw['x_voxels'].aggregate(['mean',lambda x: x.std(ddof=0)],axis=1)
        y = self.norm_embryo_raw['y_voxels'].aggregate(['mean',lambda x: x.std(ddof=0)],axis=1)
        z = self.norm_embryo_raw['z_voxels'].aggregate(['mean',lambda x: x.std(ddof=0)],axis=1)

        avg = np.hstack((
            np.array(x['mean']).reshape((len(x),1)),
            np.array(y['mean']).reshape((len(y),1)),
            np.array(z['mean']).reshape((len(z),1)),
            ))
        avg = pd.DataFrame(data=avg,index=embryo_lineage_names,columns=['x mean','y mean','z mean']).sort_index()
        self.norm_embryo = avg.round(4)

        ##### INITIALIZE POST TWITCHING LINEAGE COORDINATE INFORMATION #####
        
        for strain in self.strains:

            grouped = self.summary.groupby('lineage').get_group(strain)

            for i in range(len(grouped)):

                t0 = grouped.iloc[i,2]

                file = grouped.iloc[i,1].replace("\\RegB",f"\\RegB\\Decon_reg_{t0}\\Decon_reg_{t0}_results\\straightened_annotations\\straightened_annotations.csv")
                self.filepaths.append(f"{file}")

                img = grouped.iloc[i,1].replace("\\RegB",f"\\RegB\\Decon_reg_{t0}\\Decon_reg_{t0}_results\\output_images\\Decon_reg_{t0}_straight.tif")
                self.images.append(f"{img}")

                seam = grouped.iloc[i,1].replace("\\RegB",f"\\RegB\\Decon_reg_{t0}\\Decon_reg_{t0}_results\\straightened_lattice\\straightened_lattice.csv")
                seam = pd.read_csv(f"{seam}",index_col='name')[['x_voxels','y_voxels','z_voxels']]
                seam = seam.drop([x for x in seam.index if x.startswith('a')])
                self.seam_cells.setdefault(f"{strain}_{i}",seam)

                img = io.imread(img)
                x_bounds,y_bounds=img[0].shape
                z_bounds = len(img)

                print(strain,i,(x_bounds,y_bounds,z_bounds))
         
                file_key = grouped.iloc[i,1].replace("\\RegB","\\CellKey.csv")
                cell_key_df = pd.read_csv(file_key).iloc[0:,0:2].dropna(axis=0).map(lower)
                cell_key_df = cell_key_df[cell_key_df.map(is_digit) == True].dropna()

                self.cellkeys.setdefault(f"{strain}_{i}",cell_key_df)

                data = pd.read_csv(f"{file}")[['name','x_voxels','y_voxels','z_voxels']].dropna(axis=1,thresh=1).dropna(axis=0,thresh=1)
                self.data.setdefault(f"{strain}_{i}",data.map(lower))

                l = data.copy().map(lower)
                r = cell_key_df.copy().map(lower)

                if list(r.iloc[:,0]) == list(r.iloc[:,1]):

                    self.cellnames.setdefault(f"{strain}_{i}",l)
                    self.strain_index.append(f"{strain}_{i}")

                    norm = l.copy()
                    norm[['x_voxels','y_voxels','z_voxels']] = norm[['x_voxels','y_voxels','z_voxels']].astype(np.float64)
                    norm['x_voxels'] = norm['x_voxels'].apply(normalize,_min=0,_max=int(x_bounds))
                    norm['y_voxels'] = norm['y_voxels'].apply(normalize,_min=0,_max=int(y_bounds))
                    norm['z_voxels'] = norm['z_voxels'].apply(normalize,_min=0,_max=int(z_bounds))

                    self.normalized.setdefault(f"{strain}_{i}",norm)

                    seam_n = seam.copy()
                    seam_n[['x_voxels','y_voxels','z_voxels']] = seam_n[['x_voxels','y_voxels','z_voxels']].astype(np.float64).copy()
                    seam_n['x_voxels'] = seam_n['x_voxels'].apply(normalize,_min=0,_max=int(x_bounds))
                    seam_n['y_voxels'] = 0.5
                    seam_n['z_voxels'] = seam_n['z_voxels'].apply(normalize,_min=0,_max=int(z_bounds))
                    self.norm_seam_cells.setdefault(f"{strain}_{i}",seam_n)

                    continue
                    
                else:
                    m = r.set_index(r.iloc[:,0]).iloc[:,1]
                    l['name'] = l['name'].map(m).fillna(l['name'])

                    self.cellnames.setdefault(f"{strain}_{i}",l)
                    self.strain_index.append(f"{strain}_{i}")

                    norm = l.copy()
                    norm[['x_voxels','y_voxels','z_voxels']] = norm[['x_voxels','y_voxels','z_voxels']].astype(np.float64)
                    norm['x_voxels'] = norm['x_voxels'].apply(normalize,_min=0,_max=int(x_bounds))
                    norm['y_voxels'] = norm['y_voxels'].apply(normalize,_min=0,_max=int(y_bounds))
                    norm['z_voxels'] = norm['z_voxels'].apply(normalize,_min=0,_max=int(z_bounds))
    
                    self.normalized.setdefault(f"{strain}_{i}",norm)

                    seam_n = seam.copy()
                    seam_n[['x_voxels','y_voxels','z_voxels']] = seam_n[['x_voxels','y_voxels','z_voxels']].astype(np.float64).copy()
                    seam_n['x_voxels'] = seam_n['x_voxels'].apply(normalize,_min=0,_max=int(x_bounds))
                    seam_n['y_voxels'] = 0.5
                    seam_n['z_voxels'] = seam_n['z_voxels'].apply(normalize,_min=0,_max=int(z_bounds))
                    self.norm_seam_cells.setdefault(f"{strain}_{i}",seam_n)

        self.seam_cells_lineage_key = self.lineage_key_df.set_index('Cell').filter(items=np.array(self.seam_cells['CND-1_0'].index.map(lambda x: x.lower())),axis=0).reset_index()
        self.seam_cells_lineage_key.rename({'index':'Cell'},axis=1,inplace=True)
        self.seam_cells_lineage_key['Lineage'] = self.seam_cells_lineage_key['Lineage'].map(lambda x: x[:2].upper()+x[2:])
        self.seam_cells_lineage_key['Cell'] = self.seam_cells_lineage_key['Cell'].map(lambda x: x.upper())

        for strain in self.strain_index:

            l=self.cellnames[strain].copy()
            r=self.lineage_key_df.copy()
            
            m = r.set_index(r.iloc[:,0]).iloc[:,1]
            l['name'] = l['name'].map(m).fillna(l['name'])

            l['name'] = l['name'].apply(check_prefix_suffix)
            
            self.data_by_lineage[strain] = l

            self.normalized[strain]['name'] = l['name']

        for strain in self.strains:

            group = [x for x in self.strain_index if x.startswith(f"{strain}")]
            agg_data = {}
            agg_data_normalized = {}

            # Absolute data aggregated across samples
            i = 0
            for lin in group:
                i+=1
                a = self.data_by_lineage[lin]
                a = a[a['name'].duplicated()==False]
                a = a.set_index('name')
                agg_data.setdefault(f'{i}',a)

            agg = pd.concat(objs=agg_data.values(),axis=1,join='outer')
            agg = agg.drop([x for x in agg.index if not x.startswith(('ab','ms','cp','ca','dp','da','ea','ep','el','er'))]).astype(float)
            agg = agg.dropna(axis=0,thresh=4)
            agg = agg.rename(mapper=check_case)
            self.aggregate_data.setdefault(strain,agg.sort_index())
            
            x = agg['x_voxels'].aggregate(['mean',lambda x: x.std(ddof=0)],axis=1)
            y = agg['y_voxels'].aggregate(['mean',lambda x: x.std(ddof=0)],axis=1)
            z = agg['z_voxels'].aggregate(['mean',lambda x: x.std(ddof=0)],axis=1)

            avg = np.hstack((
                np.array(x['mean']).reshape((len(x),1)),
                np.array(y['mean']).reshape((len(y),1)),
                np.array(z['mean']).reshape((len(z),1)),
                ))
            avg = pd.DataFrame(data=avg,index=agg.index,columns=['x mean','y mean','z mean']).sort_index()
            self.avg.setdefault(strain,avg.round(1))

            stdev = np.hstack((
                np.array(x['<lambda>']).reshape((len(x),1)),
                np.array(y['<lambda>']).reshape((len(y),1)),
                np.array(z['<lambda>']).reshape((len(z),1)),
                ))
            stdev = pd.DataFrame(data=stdev,index=agg.index,columns=['x stdev','y stdev','z stdev']).sort_index()
            self.stdev.setdefault(strain,stdev.round(1))

            # Normalized data aggregated across samples
            i = 0
            for lin in group:
                i+=1
                a = self.normalized[lin]
                a = a[a['name'].duplicated()==False]
                a = a.set_index('name')
                agg_data_normalized.setdefault(f'{i}',a)

            agg = pd.concat(objs=agg_data_normalized.values(),axis=1,join='outer')
            agg = agg.drop([x for x in agg.index if not x.startswith(('ab','ms','cp','ca','dp','da','ea','ep','el','er'))]).astype(float)
            agg = agg.dropna(axis=0,thresh=4)
            agg = agg.rename(mapper=check_case)
            self.aggregate_data_normalized.setdefault(strain,agg.sort_index())
            
            x = agg['x_voxels'].aggregate(['mean',lambda x: x.std(ddof=0)],axis=1)
            y = agg['y_voxels'].aggregate(['mean',lambda x: x.std(ddof=0)],axis=1)
            z = agg['z_voxels'].aggregate(['mean',lambda x: x.std(ddof=0)],axis=1)

            avg = np.hstack((
                np.array(x['mean']).reshape((len(x),1)),
                np.array(y['mean']).reshape((len(y),1)),
                np.array(z['mean']).reshape((len(z),1)),
                ))
            avg = pd.DataFrame(data=avg,index=agg.index,columns=['x mean','y mean','z mean']).sort_index()
            self.avg_normalized.setdefault(strain,avg.round(4))

            stdev = np.hstack((
                np.array(x['<lambda>']).reshape((len(x),1)),
                np.array(y['<lambda>']).reshape((len(y),1)),
                np.array(z['<lambda>']).reshape((len(z),1)),
                ))
            stdev = pd.DataFrame(data=stdev,index=agg.index,columns=['x stdev','y stdev','z stdev']).sort_index()
            self.stdev_normalized.setdefault(strain,stdev.round(4))
    
    def display_comparison(self,threshold,normalized=False):

        if normalized == True:
            data = self.avg_normalized
            emb = self.norm_embryo

        else:
            data = self.avg
            emb = self.embryo

        fig = go.Figure(
            layout=dict(
                showlegend=True,
                autosize=True
            )
        )

        all_coords = {}
        all_coords_norm = {}
        
        for strain in self.strains:

            compared = pd.concat(
                objs=[data[f'{strain}'],emb],
                axis=1,
                join='inner'
            )
            
            compared.columns = [f'x {strain}',f'y {strain}',f'z {strain}','x embryo','y embryo','z embryo']
            
            translation_vec = np.array(compared.iloc[:,:3]) - np.array(compared.iloc[:,3:])
            magnitude = np.sqrt(translation_vec[0:,0]**2 + translation_vec[0:,1]**2 + translation_vec[0:,2]**2).round(3)
            compared = compared.assign(embryo='embryo',lin = f'{strain}')
            compared['magnitude'] = magnitude
    
            customdata = np.vstack((
                compared.index,
                magnitude.round(2).astype(str)
                )
            )

            if normalized == True:
                all_coords_norm.setdefault(f'{strain}',compared.drop(['embryo','lin'],axis=1))

            else:
                all_coords.setdefault(f'{strain}',compared.drop(['embryo','lin'],axis=1))

            
            fig.add_trace(
                go.Scatter3d(
                    x=compared['z embryo'],
                    y=compared['x embryo'],
                    z=compared['y embryo'],
                    name=f'Embryo {strain}',
                    legendgroup=f'{strain}',
                    mode='markers',
                    marker=dict(
                        size=6,
                        color=compared['z embryo'],
                        colorscale=px.colors.qualitative.Dark2,
                        opacity=1
                    ),
                    hoverinfo=['x+y+z+text'],
                    hovertext='<b>'+customdata[0,:]+' Embryo </b><br>'+'<b>Displacement: '+customdata[1,:]+'</b>',
                )
            )
            
            fig.add_trace(    
                go.Scatter3d(
                    x=compared[f'z {strain}'],
                    y=compared[f'x {strain}'],
                    z=compared[f'y {strain}'],
                    name=f'{strain}',
                    legendgroup=f'{strain}',
                    mode='markers',
                    marker=dict(
                        size=6,
                        color=compared[f'z {strain}'],
                        colorscale=px.colors.qualitative.Pastel2,
                        opacity=1
                    ),
                    hoverinfo=['x+y+z+text'],
                    hovertext='<b>'+customdata[0,:]+'</b><br>'+'<b>Displacement: '+customdata[1,:]+'</b>',
                )
            )

            for i in range(len(compared)):

                if compared.iloc[i,8] > threshold:
                    line_color='red'

                else:
                    line_color='white'

                fig.add_trace(
                    go.Scatter3d(
                        x=[compared.iloc[i,2],compared.iloc[i,5]],
                        y=[compared.iloc[i,0],compared.iloc[i,3]],
                        z=[compared.iloc[i,1],compared.iloc[i,4]],
                        mode='lines',
                        name=f"{strain}",
                        legendgroup=f'{strain}',
                        line_color=line_color,
                        line_dash='dot',
                        line_width=5,
                        hoverinfo='none',
                        showlegend=False
                    )
                )
                
        if normalized == True:
            
            fig.update_scenes(
                aspectmode='manual',
                aspectratio={'x':3,'y':1,'z':1},
                camera = {"projection": {"type": "orthographic"}},
                xaxis_title_text='x',  
                yaxis_title_text='y',  
                zaxis_title_text='z',
                xaxis={'range':[0,1]},
                yaxis={'range':[0,1]},
                zaxis={'range':[0,1]},
            )

            title_subtitle_text='Data normalized to image boundaries'
        else:
            
            fig.update_scenes(
                aspectmode='manual',
                aspectratio={'x':3,'y':1,'z':1},
                camera = {"projection": {"type": "orthographic"}},
                xaxis_title_text='x',  
                yaxis_title_text='y',  
                zaxis_title_text='z',
                xaxis={'range':[0,600]},
                yaxis={'range':[0,200]},
                zaxis={'range':[0,200]},
            )

            title_subtitle_text=None
            
        fig.update_layout(
            margin={'t':50,'b':5,'l':5,'r':5},
            template='plotly_dark',
            title_text='Pre-twitching to Post-twitching Nucleus Position Comparison',
            title_subtitle_text=title_subtitle_text
        )

        self.all_coords = all_coords
        self.all_coords_norm = all_coords_norm
        
        return fig

    def display_strain(self,strain,normalized=False):

        if normalized == True:
            data = self.aggregate_data_normalized[strain]

            a = np.array(self.norm_seam_cells[f"{strain}_0"])
            b = np.array(self.norm_seam_cells[f"{strain}_1"])
            c = np.array(self.norm_seam_cells[f"{strain}_2"])

        else:
            data = self.aggregate_data[strain]
    
            a = np.array(self.seam_cells[f"{strain}_0"])
            b = np.array(self.seam_cells[f"{strain}_1"])
            c = np.array(self.seam_cells[f"{strain}_2"])

        fig = go.Figure(
                layout=dict(
                    showlegend=True,
                    autosize=True
                )
            )
        
        customdata=np.array(['',f'{strain}_1',f'{strain}_2',f'{strain}_0'])

        traces = []
        seam_L = []
        seam_R = []
        rungs = []
        seam_names_L = np.array(self.seam_cells[f"{strain}_0"].index)[0::2]
        seam_names_R = np.array(self.seam_cells[f"{strain}_0"].index)[1::2]
        colorscale_strains = px.colors.qualitative.Vivid
        colorscale_seam_cells = px.colors.qualitative.Plotly

        for r in data.iterrows():
            
            name = r[0]
            strain_0 = np.array(r[1][0:3])
            strain_1 = np.array(r[1][3:6])
            strain_2 = np.array(r[1][6:9])

            x=[strain_0[2],strain_1[2],strain_2[2],strain_0[2]]
            y=[strain_0[0],strain_1[0],strain_2[0],strain_0[0]]
            z=[strain_0[1],strain_1[1],strain_2[1],strain_0[1]]

            trace = go.Scatter3d(
                x=x,
                y=y,
                z=z,
                name=name,
                customdata=customdata,
                hovertext=customdata,
                marker=dict(
                    size=6,
                    color=px.colors.qualitative.Plotly,
                ),
                line_color='white'
            )

            traces.append(trace)

        for i,arr in enumerate([a,b,c]):
            
            xL = arr[:,0][::2]
            yL = arr[:,1][::2]
            zL = arr[:,2][::2]

            xR = arr[:,0][1::2]
            yR = arr[:,1][1::2]
            zR = arr[:,2][1::2]

            trace_L = go.Scatter3d(
                x=zL,
                y=xL,
                z=yL,
                name=f"{strain}_{i}",
                legendgroup = f"{strain}_{i}",
                customdata=seam_names_L,
                hovertext=seam_names_L,
                mode="lines+markers+text",
                text=seam_names_L+f"_{i}",
                textfont_color=colorscale_seam_cells,
                marker=dict(
                    size=8,
                    color=colorscale_seam_cells,
                ),
                line=dict(
                    color='purple',
                )
            )

            trace_R = go.Scatter3d(
                x=zR,
                y=xR,
                z=yR,
                name=f"{strain}_{i}_R",
                legendgroup = f"{strain}_{i}",
                showlegend=False,
                customdata=seam_names_R,
                hovertext=seam_names_R,
                mode="lines+markers+text",
                text=seam_names_R+f"_{i}",
                textfont_color=colorscale_seam_cells,
                marker=dict(
                    size=8,
                    color=colorscale_seam_cells,
                ),
                line=dict(
                    color='lightgreen',
                )
            )

            seam_L.append(trace_L)
            seam_R.append(trace_R)

            for idx in range(len(xL)):
 
                rung = go.Scatter3d(
                    x=[zL[idx],zR[idx]],
                    y=[xL[idx],xR[idx]],
                    z=[yL[idx],yR[idx]],
                    name=f"{strain}_{i}",
                    legendgroup=f"{strain}_{i}",
                    showlegend=False,
                    hoverinfo=None,
                    mode='lines',
                    line_color=colorscale_seam_cells[idx],
                    line_dash='dash'
                )
    
                rungs.append(rung)

        fig.add_traces(traces)
        fig.add_traces(seam_L)
        fig.add_traces(seam_R)
        fig.add_traces(rungs)

        if normalized == True:

            fig.update_scenes(
                aspectmode='manual',
                aspectratio={'x':3,'y':1,'z':1},
                camera = {"projection": {"type": "orthographic"}},
                xaxis_title_text='x',  
                yaxis_title_text='y',  
                zaxis_title_text='z',
                xaxis={'range':[0,1]},
                yaxis={'range':[0,1]},
                zaxis={'range':[0,1]},
            )

            title_subtitle_text='Data normalized to image boundaries'
            
        else:
            fig.update_scenes(
                aspectmode='manual',
                aspectratio={'x':3,'y':1,'z':1},
                camera = {"projection": {"type": "orthographic"}},
                xaxis_title_text='x',  
                yaxis_title_text='y',  
                zaxis_title_text='z',
                xaxis={'range':[0,600]},
                yaxis={'range':[0,200]},
                zaxis={'range':[0,200]},
            )

            title_subtitle_text=None
        
        fig.update_layout(
            margin={'t':35,'b':5,'l':5,'r':5},
            template='plotly_dark',
            title_text=f"{strain} tracked data",
            title_subtitle_text=title_subtitle_text
        )

        return fig

    def display_raw_embryo(self,normalized=False,mask=None):

        if normalized == True:
            data=self.norm_embryo_raw

            a = np.array(self.norm_embryo_seam_cells['Embryo_0'])
            b = np.array(self.norm_embryo_seam_cells['Embryo_1'])
            c = np.array(self.norm_embryo_seam_cells['Embryo_2'])

        else:
            data=self.embryo_raw
            
            a = np.array(self.embryo_seam_cells['Embryo_0'])
            b = np.array(self.embryo_seam_cells['Embryo_1'])
            c = np.array(self.embryo_seam_cells['Embryo_2'])

        fig = go.Figure(
                    layout=dict(
                        showlegend=True,
                        autosize=True
                    )
                )

        customdata=np.array(['','Embryo 1','Embryo 2','Embryo 0'])

        traces = []
        seam_L = []
        seam_R = []
        rungs = []
        
        seam_names_L = np.array(self.seam_cells_lineage_key['Cell'])[0::2]
        seam_names_R = np.array(self.seam_cells_lineage_key['Cell'])[1::2]

        colorscale_strains = px.colors.qualitative.Vivid
        colorscale_seam_cells = px.colors.qualitative.Plotly

        if mask == None:
            
            for r in data.iterrows():
                
                name = r[0]
                embryo_1 = np.array(r[1][0:3])
                embryo_2 = np.array(r[1][3:6])
                embryo_3 = np.array(r[1][6:9])
    
                x=[embryo_1[2],embryo_2[2],embryo_3[2],embryo_1[2]]
                y=[embryo_1[0],embryo_2[0],embryo_3[0],embryo_1[0]]
                z=[embryo_1[1],embryo_2[1],embryo_3[1],embryo_1[1]]
    
                trace = go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    name=name,
                    customdata=customdata,
                    hovertext=customdata,
                    marker=dict(
                        size=6,
                        color=px.colors.qualitative.Plotly,
                    ),
                    line_color='white'
                )
    
                traces.append(trace)
    
            fig.add_traces(traces)

            title_suffix = ''

        elif mask != None:

            items = self.aggregate_data[mask].index

            data = data.filter(items=items,axis=0)

            for r in data.iterrows():
                
                name = r[0]
                embryo_1 = np.array(r[1][0:3])
                embryo_2 = np.array(r[1][3:6])
                embryo_3 = np.array(r[1][6:9])
    
                x=[embryo_1[2],embryo_2[2],embryo_3[2],embryo_1[2]]
                y=[embryo_1[0],embryo_2[0],embryo_3[0],embryo_1[0]]
                z=[embryo_1[1],embryo_2[1],embryo_3[1],embryo_1[1]]
    
                trace = go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    name=name,
                    customdata=customdata,
                    hovertext=customdata,
                    marker=dict(
                        size=6,
                        color=px.colors.qualitative.Plotly,
                    ),
                    line_color='white'
                )
    
                traces.append(trace)
    
            fig.add_traces(traces)
            
            title_suffix = f' masked by {mask} dataset'

        for i,arr in enumerate([a,b,c]):

            xL = arr[:,0][::2]
            yL = arr[:,1][::2]
            zL = arr[:,2][::2]

            xR = arr[:,0][1::2]
            yR = arr[:,1][1::2]
            zR = arr[:,2][1::2]

            trace_L = go.Scatter3d(
                x=zL,
                y=xL,
                z=yL,
                name=f"Embryo_{i}",
                legendgroup = f"Embryo_{i}",
                customdata=seam_names_L,
                hovertext=seam_names_L,
                mode="lines+markers+text",
                text=seam_names_L+f"_{i}",
                textfont_color=colorscale_seam_cells,
                marker=dict(
                    size=8,
                    color=colorscale_seam_cells,
                ),
                line=dict(
                    color='purple',
                )
            )

            trace_R = go.Scatter3d(
                x=zR,
                y=xR,
                z=yR,
                name=f"Embryo_{i}",
                legendgroup = f"Embryo_{i}",
                showlegend=False,
                customdata=seam_names_R,
                hovertext=seam_names_R,
                mode="lines+markers+text",
                text=seam_names_R+f"_{i}",
                textfont_color=colorscale_seam_cells,
                marker=dict(
                    size=8,
                    color=colorscale_seam_cells,
                ),
                line=dict(
                    color='lightgreen',
                )
            )

            for idx in range(len(xL)):
 
                rung = go.Scatter3d(
                    x=[zL[idx],zR[idx]],
                    y=[xL[idx],xR[idx]],
                    z=[yL[idx],yR[idx]],
                    name=f"Embryo_{i}",
                    legendgroup=f"Embryo_{i}",
                    hoverinfo=None,
                    showlegend=False,
                    mode='lines',
                    line_color=colorscale_seam_cells[idx],
                    line_dash='dash'
                )
    
                rungs.append(rung)

            seam_L.append(trace_L)
            seam_R.append(trace_R)

        fig.add_traces(seam_L)
        fig.add_traces(seam_R)
        fig.add_traces(rungs)

        if normalized == True:
            
            fig.update_scenes(
                aspectmode='manual',
                aspectratio={'x':3,'y':1,'z':1},
                camera = {"projection": {"type": "orthographic"}},
                xaxis_title_text='x',  
                yaxis_title_text='y',  
                zaxis_title_text='z',
                xaxis={'range':[0,1]},
                yaxis={'range':[0,1]},
                zaxis={'range':[0,1]},
            )

            title_subtitle_text='Data normalized to image boundaries'

        else:
            
            fig.update_scenes(
                aspectmode='manual',
                aspectratio={'x':3,'y':1,'z':1},
                camera = {"projection": {"type": "orthographic"}},
                xaxis_title_text='x',  
                yaxis_title_text='y',  
                zaxis_title_text='z',
                xaxis={'range':[0,600]},
                yaxis={'range':[0,200]},
                zaxis={'range':[0,200]},
            )

            title_subtitle_text=None

        fig.update_layout(
            margin={'t':50,'b':5,'l':5,'r':5},
            template='plotly_dark',
            title_text='Pretwitch embryo data'+title_suffix,
            title_subtitle_text=title_subtitle_text
        )

        return fig

    def check_outliers(self,strain,threshold):

        if type(strain) == list:

            filtered = {}
            
            for i in strain:
                
                matrix = self.all_coords[i]
                matrix = matrix[matrix['magnitude']>threshold]
                filtered.setdefault(i,matrix)

            reformat = {}
            length = max([len(x) for x in list(filtered.values())])
            for d in filtered.items():
            
                index = np.array(d[1].index)
                mag = np.array(d[1]['magnitude'])
                cell_ids = np.pad(index,(0,length-len(index)),mode='constant',constant_values=np.nan)
                magnitude = np.pad(mag,(0,length-len(mag)),mode='constant',constant_values=np.nan)
            
                reformat.setdefault((d[0],'cell id'),cell_ids)
                reformat.setdefault((d[0],'magnitude'),magnitude)

            return pd.DataFrame(reformat)

        else:
            matrix = self.all_coords[strain]
            matrix = matrix[matrix['magnitude']>threshold]

            return matrix

    def display_cellkey(self,strain,normalized=False):
        
        def lower(string):
            string = str(string)
            string = string.lower()
            return string
            
        def check_case(string):

            if string.isdigit() == True:
                pass
            elif string.startswith(('ca','cp','d','ep','ea')):
                string = string[:1].upper() + string[1:].lower()
                return string
            elif string.startswith(('ab','ms','dd')):
                string = string[:2].upper() + string[2:].lower()
                return string
            else:
                return string
                
        strain_group = strain[0:-2]
        strain_idx = strain[-2:]
        
        if strain_idx == '_0':
            start = 0
            end = 3
        elif strain_idx == '_1':
            start = 3
            end = 6
        elif strain_idx == '_2':
            start = 6
            end = 9

        l = self.avg[strain_group].copy().reset_index().map(lower)
        k = self.cellnames[strain].copy()
        r = self.cellkeys[strain].copy()
        ids = self.lineage_key_df

        m = ids.set_index(ids.iloc[:,0]).iloc[:,1]
        r.iloc[:,1] = r.iloc[:,1].map(m).fillna(r.iloc[:,1])

        m= r.set_index(r.iloc[:,1]).iloc[:,0]
        l['cellkey'] = l['name'].map(m).fillna(l['name'])

        l['name'] = l['name'].map(check_case)

        l.rename(columns={'x mean':'x','y mean':'y','z mean':'z'},inplace=True)
        
        l[['y','z','x']] = np.array(self.aggregate_data[strain_group])[:,start:end]

        if normalized == True:

            l[['y','z','x']] = np.array(self.aggregate_data_normalized[strain_group])[:,start:end]
            
        return l
