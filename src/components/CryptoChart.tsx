import React from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  ChartOptions
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

interface CryptoChartProps {
  data: number[];
  labels: string[];
  timeframe: string;
  darkMode: boolean;
}

const CryptoChart: React.FC<CryptoChartProps> = ({ data, labels, timeframe, darkMode }) => {
  const chartData = {
    labels,
    datasets: [
      {
        label: `Price (${timeframe})`,
        data,
        borderColor: data[0] < data[data.length - 1] ? 'rgb(34, 197, 94)' : 'rgb(239, 68, 68)',
        backgroundColor: data[0] < data[data.length - 1] ? 'rgba(34, 197, 94, 0.1)' : 'rgba(239, 68, 68, 0.1)',
        fill: true,
        tension: 0.4,
      },
    ],
  };

  const options: ChartOptions<'line'> = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top' as const,
        labels: {
          color: darkMode ? 'white' : 'black',
        },
      },
      title: {
        display: false,
      },
    },
    scales: {
      x: {
        ticks: {
          color: darkMode ? 'white' : 'black',
        },
        grid: {
          color: darkMode ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)',
        },
      },
      y: {
        ticks: {
          color: darkMode ? 'white' : 'black',
        },
        grid: {
          color: darkMode ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)',
        },
      },
    },
  };

  return <Line data={chartData} options={options} />;
};

export default CryptoChart;