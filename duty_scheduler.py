#!/usr/bin/env python
# coding: utf-8
# from .ipynb (jupyter nbconvert --to script) 

# In[1]:


from ortools.sat.python import cp_model
from datetime import date, datetime, timedelta
import pandas as pd
import streamlit as st
import xlsxwriter
import io


# In[2]:


# 0) Title of page

st.title('Duty Scheduler')
st.markdown('This app helps you do up a first-cut of duty schedules. :smile:')
st.caption('Prototype, Ver 1.0')


# In[3]:


class ShiftScheduler:
    def __init__(
        self,
        start_date: date,
        end_date: date,
        num_shifts: int = None,
        num_empl: int = None,
        num_people: int = None,
        rest_shifts: int = None,
        ad_hoc_off: list = None,
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.S = num_shifts
        self.E = num_empl
        self.P = num_people
        self.rest_shifts = rest_shifts
        self.ad_hoc_off = set(ad_hoc_off) or []

        # Derived number of days
        self.D = (self.end_date - self.start_date).days + 1

        # Placeholders for model & solution
        self.model = None
        self.vars = {}
        self.first_solution = None
        self.solution_count = 0

    def build_model(self):
        model = cp_model.CpModel()

        # Decision variables x[(d,s,e,p)] ∈ {0,1}
        for d in range(self.D):
            for s in range(self.S):
                for e in range(self.E):
                    for p in range(self.P):
                        self.vars[(d, s, e, p)] = model.NewBoolVar(
                            f"x_d{d}_s{s}_e{e}_p{p}"
                        )

        # 1) Exactly one person per slot (or none if ad-hoc off)
        for d in range(self.D):
            for s in range(self.S):
                for e in range(self.E):
                    if (d, s, e) in self.ad_hoc_off:
                        for p in range(self.P):
                            model.Add(self.vars[(d, s, e, p)] == 0)
                    else:
                        model.Add(
                            sum(self.vars[(d, s, e, p)] for p in range(self.P))
                            == 1
                        )

        # 2) No adjacent shifts (same day and overnight)
        for p in range(self.P):
            # same-day adjacency
            for d in range(self.D):
                for s in range(self.S - 1):
                    model.Add(
                        sum(self.vars[(d, s,   e, p)] for e in range(self.E)) +
                        sum(self.vars[(d, s+1, e, p)] for e in range(self.E))
                        <= 1
                    )
            # cross-day adjacency
            for d in range(self.D - 1):
                model.Add(
                    sum(self.vars[(d, self.S-1, e, p)] for e in range(self.E)) +
                    sum(self.vars[(d+1, 0, e, p)] for e in range(self.E))
                    <= 1
                )

        # 3) Balance shifts per person
        shifts_per_p = []
        for p in range(self.P):
            var = model.NewIntVar(0, self.D * self.S * self.E, f"shifts_p{p}")
            model.Add(
                var
                == sum(
                    self.vars[(d, s, e, p)]
                    for d in range(self.D)
                    for s in range(self.S)
                    for e in range(self.E)
                )
            )
            shifts_per_p.append(var)

        max_shifts = model.NewIntVar(0, self.D * self.S * self.E, "max_shifts")
        min_shifts = model.NewIntVar(0, self.D * self.S * self.E, "min_shifts")
        model.AddMaxEquality(max_shifts, shifts_per_p)
        model.AddMinEquality(min_shifts, shifts_per_p)
        model.Minimize(max_shifts - min_shifts)

        self.model = model

    def solve(self):
        if self.model is None:
            self.build_model()

        solver = cp_model.CpSolver()
        solver.parameters.enumerate_all_solutions = True

        # Inner callback to count & capture the first solution
        class SolutionCounter(cp_model.CpSolverSolutionCallback):
            def __init__(self, outer):
                super().__init__()
                self.outer = outer
            def on_solution_callback(self):
                self.outer.solution_count += 1
                if self.outer.solution_count == 1:
                    sol = []
                    for d in range(self.outer.D):
                        day_rec = []
                        for s in range(self.outer.S):
                            shift_rec = []
                            for e in range(self.outer.E):
                                assigned = None
                                for p in range(self.outer.P):
                                    if self.Value(self.outer.vars[(d, s, e, p)]):
                                        assigned = p
                                        break
                                shift_rec.append(assigned) # To prevent nothing from being appended, append "None"    
                            day_rec.append(shift_rec)
                        sol.append(day_rec)
                    self.outer.first_solution = sol

        counter = SolutionCounter(self)
        solver.Solve(self.model, counter)
        
        print(f"first_solution[0][0]: {self.first_solution[0][0]}")
    
    def get_schedule_df(self) -> pd.DataFrame:
        """Returns a DataFrame with rows=shifts (0…S-1) and
           columns as a MultiIndex (date, emplacement)."""
        try:
            print("Entered get_schedule_df")

            if self.first_solution is None:
                raise ValueError("No solution found. Try adjusting the inputs.")

            records = {}
            dates = [self.start_date + timedelta(days=d) for d in range(self.D)]

            for d_idx, dt in enumerate(dates):
                for e in range(self.E):
                    col = (dt, f"empl{e}")
                    shift_vals = []
                    for s in range(self.S):
                        if (d_idx, s, e) in self.ad_hoc_off:
                            shift_vals.append("Off")
                        else:
                            try:
                                if e < len(self.first_solution[d_idx][s]):
                                    val = self.first_solution[d_idx][s][e]
                                else:
                                    print(f"Emplacement index {e} out of bounds for day={d_idx}, shift={s}")
                                    val = None
                            except (IndexError, TypeError) as err:
                                print(f"Access error at day={d_idx}, shift={s}, empl={e}: {err}")
                                val = None
                            shift_vals.append(val)
                    records[col] = shift_vals

            df1 = pd.DataFrame(records, index=pd.Index(range(self.S), name="shift"))
            df1.columns = pd.MultiIndex.from_tuples(df1.columns, names=["date", "empl"])
            return df1

        except Exception as e:
            import traceback
            traceback.print_exc()
            raise e
    
    def get_shift_counts_df(self) -> pd.DataFrame:
        """Returns a DataFrame with each person’s total assigned shifts."""
        if self.first_solution is None:
            raise ValueError("No solution found. Try adjusting the inputs.")

        counts = {p: 0 for p in range(self.P)}

        for d in range(self.D):
            for s in range(self.S):
                for e in range(self.E):
                    try:
                        p = self.first_solution[d][s][e]
                        if isinstance(p, int) and p in counts:
                            counts[p] += 1
                        else:
                            print(f"Unexpected value at d={d}, s={s}, e={e}: {p}")
                    except (IndexError, TypeError) as err:
                        print(f"Access error at d={d}, s={s}, e={e}: {err}")
                        continue

        df2 = (
            pd.DataFrame.from_dict(counts, orient="index", columns=["num_shifts"])
            .sort_index()
            .rename_axis("person")
        )
        return df2


# In[4]:


# Define inputs

# Dates

today = datetime.now()
year = today.year
jan_1 = date(year, 1, 1)
dec_31 = date(year, 12, 31)

st.subheader("Step 1: Input dates.")

date_input = st.date_input(
    "Select duty dates.",
    (jan_1, dec_31),
    min_value=jan_1,
    max_value=dec_31,
    format="MM.DD.YYYY"
)

if isinstance(date_input, tuple) and len(date_input) == 2:
    start_date_input, end_date_input = date_input
else:
    st.error("Please select a date range.")
    st.stop()
    
# Number of shifts, emplacements, people

st.subheader("Step 2: Input number of shifts, emplacements and people.")

num_shifts_input = st.selectbox("Input number of shifts per duty day.", (4,1,2,3,6,8,12))
num_empl_input = st.slider("Input number of emplacements per duty day.", 1,10,1)
num_people_input = st.number_input("Input number of people available for duty.", min_value = 10)
rest_shifts_input = st.number_input("Input minimum number of shifts required between duty shifts.", min_value = 1, max_value=num_shifts_input) 


# In[5]:


st.subheader("Step 3: Select slots where manning is not required.")

# Build a DataFrame of all slots + a 'blocked' column
D = (end_date_input - start_date_input).days + 1
dates = [start_date_input + timedelta(days=d) for d in range(D)]

records = []
for d_idx, dt in enumerate(dates):
    for s in range(num_shifts_input):
        for e in range(num_empl_input):
            records.append({
                "Day": d_idx,
                "Date": dt,
                "Shift": s,
                "Emplacement": e,
                "No Manning": False
            })

slots_df = pd.DataFrame(records)

# Render the editable table for visual selection
st.markdown("Select slots that do not need manning (tick to select).")
edited = st.data_editor(
    slots_df,
    column_config={"No Manning": st.column_config.CheckboxColumn("No Manning")},
    width="stretch"
)

# Extract the blocked slots back into tuples
ad_hoc_off = [
    (int(row["Day"]), int(row["Shift"]), int(row["Emplacement"]))
    for _, row in edited.iterrows()
    if row["No Manning"]
]

st.markdown(f"**You have selected** {len(ad_hoc_off)} slots that does not need manning.")


# In[6]:


# Execute button and process

execute = st.button("Generate output", type="primary")

if execute == True:
    scheduler = ShiftScheduler(
        start_date=start_date_input,
        end_date=end_date_input,
        num_shifts=num_shifts_input,
        num_empl=num_empl_input,
        num_people=num_people_input,
        rest_shifts=rest_shifts_input,
        ad_hoc_off=ad_hoc_off,
    )
    scheduler.ad_hoc_off = ad_hoc_off
    scheduler.build_model()
    scheduler.solve()
    df1 = scheduler.get_schedule_df()
    df2 = scheduler.get_shift_counts_df()
    
    # Display solution
    
    st.markdown(f"Total solutions found: {scheduler.solution_count}")
    st.markdown("\nSample schedule:")
    st.dataframe(df1)

    # Display number of shifts per person for the solution
    
    st.markdown("\nShifts per person:")
    st.dataframe(df2)
    
    # Display 
    
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df1.to_excel(writer, sheet_name="schedule")
        df2.to_excel(writer, sheet_name="number of shifts per person") 
        writer.close()
        data_final = buffer.getvalue()
    
    st.download_button("\nDownload output as Excel file", data=data_final, file_name="output.xlsx", mime='text/xlsx')
