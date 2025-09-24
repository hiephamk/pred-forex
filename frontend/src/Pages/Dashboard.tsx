import { Box, VStack, HStack} from "@chakra-ui/react";
import Dashboard_Nav from "@/components/NavBar/Dashboard_Nav";
import { Outlet } from "react-router";
import { useColorPalette } from "@/components/Helper/Utils";

function Dashboard() {
  const colors = useColorPalette()
  return(
    <VStack minH={"100vh"}>
      <Box mt={"10px"}>
        <Dashboard_Nav/>
      </Box>
      <HStack justifyContent={"space-between"} w={"99vw"} gap={"10px"}>
        <Box h={"fit-content"} w={"100%"}>
          <Outlet/>
        </Box>
      </HStack>
    </VStack>
  )
}

export default Dashboard;