import { useEffect, useState } from "react";
import {
  Box,
  Heading,
  Spinner,
  Table,
  VStack,
  Button,
//   useToast,
} from "@chakra-ui/react";
import { Line } from "react-chartjs-2";
import { Prediction } from "../types";

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

function Dashboard() {
  const [preds, setPreds] = useState<Prediction[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [training, setTraining] = useState<boolean>(false);
//   const toast = useToast();

  // Fetch predictions from Django API
  const fetchPreds = () => {
    setLoading(true);
    fetch("http://localhost:8000/api/forex/forecast/?days=5")
      .then((res) => res.json())
      .then((data) => {
        setPreds(data);
        setLoading(false);
      })
      .catch(() => {
        setLoading(false);
        // toast({
        //   title: "Error",
        //   description: "Failed to fetch predictions.",
        //   status: "error",
        //   duration: 5000,
        //   isClosable: true,
        // });
      });
  };

  // Trigger model training
  const trainModel = () => {
    setTraining(true);
    fetch("http://localhost:8000/api/forex/train_nn/", { method: "POST" })
      .then((res) => res.json())
      .then((data) => {
        alert("Training Complete")
        // toast({
        //   title: "Training Complete",
        //   description: data.status,
        //   status: "success",
        //   duration: 5000,
        //   isClosable: true,
        // });
        fetchPreds(); // Refresh predictions after training
      })
      .catch(() => {
        alert("Error")
        // toast({
        //   title: "Error",
        //   description: "Failed to train model.",
        //   status: "error",
        //   duration: 5000,
        //   isClosable: true,
        // });
      })
      .finally(() => setTraining(false));
  };

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
    <Box maxW="container.lg" mx="auto" py={10}>
      <VStack gap={8}>
        <Heading size="lg" textAlign="center">
          PyTorch NN Forecast (XAU/USD)
        </Heading>

        {/* Chart */}
        <Box w="full" p={4} boxShadow="md" borderWidth="1px" rounded="lg">
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

export default Dashboard;