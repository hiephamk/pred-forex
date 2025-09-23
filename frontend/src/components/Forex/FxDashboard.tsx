import { useEffect, useState } from "react";
import axios from "axios";
import {
  Box,
  Heading,
  Spinner,
  Table,
  VStack,
  HStack,
  Button,
//   useToast,
} from "@chakra-ui/react";
import { Line } from "react-chartjs-2";

import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  LineElement,
  PointElement,
  Tooltip,
  Legend,
} from "chart.js";

ChartJS.register(CategoryScale, LinearScale, LineElement, PointElement, Tooltip, Legend);

type Prediction = {
  date: string;
  predicted_close: number;
};

function FxDashboard() {
  const [preds, setPreds] = useState<Prediction[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [training, setTraining] = useState<boolean>(false);
  

 const fetchPreds = async () => {
    setLoading(true);
    const url = "http://localhost:8000/api/forex/forecast/?days=5"
    try {
        const res = await axios.get(url)
        setPreds(res.data)
    }catch(error) {
        console.error('fetch pred erros', error)
    }finally {
        setLoading(false);
    }
 }
  // Trigger model training

  const trainModel = async() => {
    setTraining(true)
    const url = "http://localhost:8000/api/forex/train_nn/"
    try {
        await axios.post(url)
        alert("Training complete")
    }catch(error){
        alert("training error")
        console.error("training error", error)
    }finally{
        setTraining(false)
    }
  }

  useEffect(() => {
    fetchPreds();
  }, []);

  if (loading || training) {
    return (
      <Box display="flex" justifyContent="center" mt={20}>
        <Spinner size="xl" />
      </Box>
    );
  }

  const chartData = {
    labels: preds.map((p) => p.date),
    datasets: [
      {
        label: "Predicted Close",
        data: preds.map((p) => p.predicted_close),
        borderColor: "gold",
        backgroundColor: "rgba(255, 215, 0, 0.5)",
        fill: true,
        tension: 0.3,
      },
    ],
  };

  return (
    <Box mx="auto" py={10}>
      <VStack gap={8}>
        <Heading size="lg" textAlign="center">
          Forex Forecast (XAU/USD)
        </Heading>

        {/* Chart */}
        <HStack w={"50%"}>
          <Box h={"auto"} w="full" p={4} boxShadow="md" borderWidth="1px" rounded="lg">
            <Line data={chartData} />
          </Box>
          {/* Table */}
          <Box w="full" boxShadow="md" borderWidth="1px" rounded="lg" p={4}>
            <Table.Root>
              <Table.Header>
                <Table.Row>
                  <Table.ColumnHeader>Date</Table.ColumnHeader>
                  <Table.ColumnHeader textAlign="end">Predicted Close</Table.ColumnHeader>
                </Table.Row>
              </Table.Header>
              <Table.Body>
                {preds.map((p, i) => (
                  <Table.Row key={i}>
                    <Table.Cell>{p.date}</Table.Cell>
                    <Table.Cell textAlign="end">{p.predicted_close.toFixed(2)}</Table.Cell>
                  </Table.Row>
                ))}
              </Table.Body>
            </Table.Root>
          </Box>
        </HStack>

        {/* Buttons */}
        <VStack gap={4}>
          <Button bg={"black"}
            onClick={fetchPreds}
            // isLoading={loading}
            loadingText="Refreshing"
          >
            Refresh Predictions
          </Button>
          <Button bg={"black"}
            onClick={trainModel}
            // isLoading={training}
            loadingText="Training"
          >
            Train Model
          </Button>
        </VStack>
      </VStack>
    </Box>
  );
}

export default FxDashboard;