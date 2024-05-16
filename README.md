# JBI100: UK Road Safety Data Visualization

This project is a web-based interactive visualization of road safety data in Great Britain. It was developed as part of the JBI100 Visualization course resit, utilizing the [Road Safety Data](https://www.data.gov.uk/dataset/cb7ae6f0-4be6-4935-9277-47e5ce24a11f/road-safety-data) dataset from data.gov.uk.

## Features

### Interactive Map

- The main view of the application is an interactive map displaying accident locations across Great Britain.
- Each data point on the map represents an accident, color-coded by its severity (Fatal, Serious, or Slight).
- Hovering over a data point reveals detailed information about the accident, including the date, time, driver's sex, age, vehicle age, light conditions, and vehicle type.

### Filtering Options

The left panel of the application provides various filtering options to customize the displayed data:

- **Date Range**: Select a specific range of dates to focus on accidents that occurred within that period.
- **Time of Day**: Filter accidents based on the time of day they occurred, allowing for analysis of patterns related to different times.
- **Driver Age**: Set a range for the age of the driver involved in the accidents to examine the relationship between driver age and accident occurrence.
- **Vehicle Age**: Choose a range for the age of the vehicles involved in the accidents to explore potential correlations.
- **Light Conditions**: Filter accidents based on the light conditions at the time of the incident (e.g., daylight, darkness).
- **Accident Severity**: Select specific severity levels (Fatal, Serious, Slight) to focus on accidents of particular interest.
- **Driver Gender**: Filter accidents based on the gender of the driver involved.

### Data Visualization

The right panel of the application presents various charts and visualizations to provide insights into the filtered data:

- **Accident Severity by Driver Gender**: A pie chart displaying the distribution of accident severity for each driver gender category.
- **Driver Age Distribution**: A bar plot with a kernel density estimation curve showing the distribution of driver ages involved in the filtered accidents.
- **Vehicle Age Distribution**: A bar plot with a kernel density estimation curve presenting the distribution of vehicle ages involved in the filtered accidents.
- **First Point of Impact vs. Accident Severity**: A heatmap illustrating the relationship between the first point of impact and accident severity, helping to identify patterns and critical impact points.

### Interactivity

- Selecting data points on the map updates the charts and heatmap in the right panel to reflect the characteristics of the selected accidents.
- The charts and heatmap are dynamically updated based on the selected filters, allowing for real-time exploration and analysis of the data.

## Dataset

The visualization is based on the [Road Safety Data](https://www.data.gov.uk/dataset/cb7ae6f0-4be6-4935-9277-47e5ce24a11f/road-safety-data) dataset from data.gov.uk, which contains information about road accidents in Great Britain. The dataset includes details such as accident location, date, time, driver information, vehicle information, and accident severity.

## Purpose

The purpose of this visualization is to provide an interactive and intuitive way to explore and analyze road safety data in Great Britain. By offering various filtering options and presenting the data through different visualizations, the application aims to facilitate a deeper understanding of the factors contributing to road accidents and their severity. This insight can be valuable for policymakers, researchers, and road safety organizations in identifying patterns, trends, and areas for improvement in road safety measures.