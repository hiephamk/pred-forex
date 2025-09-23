
import { Box, VStack, HStack} from "@chakra-ui/react";
import Dashboard_Nav from "@/components/NavBar/Dashboard_Nav";
import { Outlet } from "react-router";

function Dashboard() {
  return(
    <VStack minH={"100vh"}>
      <Dashboard_Nav/>
      <HStack justifyContent={"space-between"} w={"99vw"} gap={"10px"}>
        <Box h={"fit-content"} w={"100%"} rounded={"7px"}>
          <Outlet/>
        </Box>
      </HStack>
    </VStack>

  )
}

export default Dashboard;